import time

import openai
import pymongo

# MongoDB Atlas connection URI (replace with your connection string)

# client = pymongo.MongoClient("mongodb://localhost:32768/?directConnection=true")
client = pymongo.MongoClient("mongodb://localhost:49774/?directConnection=true")
DATABASE_NAME = "kb"
db = client[DATABASE_NAME]
openai.api_key = ""


def get_openai_embedding(text, retries=5, delay=5):
    """
    Get OpenAI embedding with retry logic.

    Parameters:
    - text: The text to get embeddings for.
    - retries: Number of retry attempts.
    - delay: Delay between retries in seconds.

    Returns:
    - embedding: The embedding vector.
    """
    for attempt in range(retries):
        try:
            response = openai.embeddings.create(
                input=text, model="text-embedding-ada-002"
            )
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            print(f"Error fetching embedding: {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise  # Re-raise the exception if retries are exhausted


def get_vector_search(hybrid_query_vector, hybrid_query, max_sources):
    vector_index = "content_embedding_vector_cosine"
    embedding_path = "content_embedding"
    COLLECTION_NAME = "knowledge_base"
    collection = db[COLLECTION_NAME]
    number_of_docs = collection.count_documents({})
    pipeline_test = [
        {
            "$vectorSearch": {
                "index": vector_index,
                "path": embedding_path,
                "queryVector": hybrid_query_vector,
                "exact": True,  # exact match ENN # no change
                "limit": number_of_docs,
            }
        },
        {
            "$addFields": {
                "cosine_score": {
                    "$subtract": [
                        {"$multiply": [{"$meta": "vectorSearchScore"}, 2]},
                        1,
                    ]
                }
            }
        },
        {"$limit": max_sources},  # Limit to the top 3 unique `id`s
        {
            "$project": {
                "_id": 1,
                "articleId": 1,
                "id": 1,
                "html_url": 1,
                # "content": 1,
                "title": 1,
                "cosine_score": 1,
            }
        },
    ]

    # run pipeline
    result = list(collection.aggregate(pipeline_test))

    return result


def get_hybrid_search(hybrid_query_vector, hybrid_query, max_sources):
    vector_index = "content_embedding_vector_cosine"
    embedding_path = "content_embedding"
    COLLECTION_NAME = "knowledge_base"
    collection = db[COLLECTION_NAME]
    number_of_docs = collection.count_documents({})
    vector_weight = 0.5
    full_text_weight = 0.5
    text_index = "content_index"
    text_param = "content"
    pipeline_test = [
        {
            "$vectorSearch": {
                "index": vector_index,
                "path": embedding_path,
                "queryVector": hybrid_query_vector,
                "exact": True,
                "limit": number_of_docs,
            }
        },
        {
            "$addFields": {
                "cosine_score": {
                    "$subtract": [
                        {"$multiply": [{"$meta": "vectorSearchScore"}, 2]},
                        1,
                    ]
                }
            }
        },
        {"$group": {"_id": None, "docs": {"$push": "$$ROOT"}}},
        {"$unwind": {"path": "$docs", "includeArrayIndex": "rank"}},
        {
            "$addFields": {
                "vs_score": {
                    "$multiply": [
                        vector_weight,
                        {"$divide": [1.0, {"$add": ["$rank", 1]}]},
                    ]
                }
            }
        },
        {
            "$project": {
                "vs_score": 1,
                "cosine_score": "$docs.cosine_score",
                "_id": "$docs._id",
                "id": "$docs.id",
                "articleId": "$docs.articleId",
                "content": "$docs.content",
                "html_url": "$docs.html_url",
                "title": "$docs.title",
            }
        },
        {
            "$unionWith": {
                "coll": COLLECTION_NAME,
                "pipeline": [
                    {
                        "$search": {
                            "index": text_index,
                            "text": {
                                "query": hybrid_query,  # Add the query text here
                                "path": text_param,
                            },
                        }
                    },
                    {"$limit": number_of_docs},
                    {"$group": {"_id": None, "docs": {"$push": "$$ROOT"}}},
                    {"$unwind": {"path": "$docs", "includeArrayIndex": "rank"}},
                    {
                        "$addFields": {
                            "fts_score": {
                                "$multiply": [
                                    full_text_weight,
                                    {"$divide": [1.0, {"$add": ["$rank", 1]}]},
                                ]
                            }
                        }
                    },
                    {
                        "$project": {
                            "fts_score": 1,
                            "_id": "$docs._id",
                            "id": "$docs.id",
                            "articleId": "$docs.articleId",
                            "content": "$docs.content",
                            "html_url": "$docs.html_url",
                            "title": "$docs.title",
                        }
                    },
                ],
            }
        },
        {
            "$group": {
                "_id": "$_id",
                "id": {"$first": "$id"},
                "articleId": {"$first": "$articleId"},
                "content": {"$first": "$content"},
                "html_url": {"$first": "$html_url"},
                "title": {"$first": "$title"},
                "vs_score": {"$max": "$vs_score"},
                "fts_score": {"$max": "$fts_score"},
                "cosine_score": {"$max": "$cosine_score"},
            }
        },
        {
            "$group": {
                "_id": "$id",  # Group by `id` field
                "bestMatch": {
                    "$first": "$$ROOT"
                },  # Take the highest score (first document after sorting)
            }
        },
        {
            "$replaceRoot": {
                "newRoot": "$bestMatch"  # Replace root with the best matching document for each group
            }
        },
        {
            "$project": {
                "_id": 1,
                "id": 1,
                "articleId": 1,
                "content": 1,
                "html_url": 1,
                "title": 1,
                "vs_score": {"$ifNull": ["$vs_score", 0]},
                "fts_score": {"$ifNull": ["$fts_score", 0]},
                "cosine_score": {"$ifNull": ["$cosine_score", 0]},
            }
        },
        {
            "$project": {
                "score": {"$add": ["$fts_score", "$vs_score"]},
                "_id": 1,
                "id": 1,
                "vs_score": 1,
                "fts_score": 1,
                "cosine_score": 1,
                "articleId": 1,
                # "content": 1,
                "html_url": 1,
                "title": 1,
            }
        },
        {"$sort": {"score": -1}},
        {"$limit": max_sources},
    ]
    # run pipeline
    result = list(collection.aggregate(pipeline_test))

    return result


def get_chunking_search(hybrid_query_vector, hybrid_query, max_sources):
    return []
    vector_index = "chunk_content_embedding_vector_cosine"
    embedding_path = "chunk_content_embedding"
    COLLECTION_NAME = "knowledge_base_chunking"
    collection = db[COLLECTION_NAME]
    number_of_docs = collection.count_documents({})
    pipeline_test = [
        {
            "$vectorSearch": {
                "index": vector_index,
                "path": embedding_path,
                "queryVector": hybrid_query_vector,
                "exact": True,  # exact match ENN # no change
                "limit": number_of_docs,  # Temporarily increase limit to collect more results
            }
        },
        {"$addFields": {"score": {"$meta": "vectorSearchScore"}}},
        {
            "$group": {
                "_id": "$id",  # Group by `id` field
                "bestMatch": {
                    "$first": "$$ROOT"
                },  # Take the highest score (first document after sorting)
            }
        },
        {
            "$replaceRoot": {
                "newRoot": "$bestMatch"  # Replace root with the best matching document for each group
            }
        },
        {
            "$sort": {"score": -1},  # Sort by similarity score in descending order
        },
        {"$limit": max_sources},  # Limit to the top 3 unique `id`s
        {
            "$project": {
                "_id": 1,
                "articleId": 1,
                "id": 1,
                "html_url": 1,
                # "content": 1,
                "title": 1,
                "score": 1,
            }
        },
    ]

    # run pipeline
    result = list(collection.aggregate(pipeline_test))

    return result


def get_hybrid_chunking_search(hybrid_query_vector, hybrid_query, max_sources):
    return []
    vector_index = "chunk_content_embedding_vector_cosine"
    embedding_path = "chunk_content_embedding"
    COLLECTION_NAME = "knowledge_base_chunking"
    collection = db[COLLECTION_NAME]
    number_of_docs = collection.count_documents({})
    vector_weight = 0.75
    full_text_weight = 0.25
    text_index = "chunk_content_index"
    text_param = "chunk_content"
    pipeline_test = [
        {
            "$vectorSearch": {
                "index": vector_index,
                "path": embedding_path,
                "queryVector": hybrid_query_vector,
                "exact": True,
                "limit": number_of_docs,
            }
        },
        {
            "$addFields": {
                "cosine_score": {
                    "$subtract": [
                        {"$multiply": [{"$meta": "vectorSearchScore"}, 2]},
                        1,
                    ]
                }
            }
        },
        {
            "$group": {
                "_id": "$id",  # Group by `id` field
                "bestMatch": {
                    "$first": "$$ROOT"
                },  # Take the highest score (first document after sorting)
            }
        },
        {
            "$replaceRoot": {
                "newRoot": "$bestMatch"  # Replace root with the best matching document for each group
            }
        },
        {
            "$sort": {
                "cosine_score": -1
            },  # Sort by similarity score in descending order
        },
        {"$group": {"_id": None, "docs": {"$push": "$$ROOT"}}},
        {"$unwind": {"path": "$docs", "includeArrayIndex": "rank"}},
        {
            "$addFields": {
                "vs_score": {
                    "$multiply": [
                        vector_weight,
                        {"$divide": [1.0, {"$add": ["$rank", 1]}]},
                    ]
                }
            }
        },
        {
            "$project": {
                "vs_score": 1,
                "cosine_score": "$docs.cosine_score",
                "_id": "$docs._id",
                "id": "$docs.id",
                "articleId": "$docs.articleId",
                "content": "$docs.content",
                "html_url": "$docs.html_url",
                "title": "$docs.title",
            }
        },
        {
            "$unionWith": {
                "coll": COLLECTION_NAME,
                "pipeline": [
                    {
                        "$search": {
                            "index": text_index,
                            "text": {
                                "query": hybrid_query,  # Add the query text here
                                "path": text_param,
                            },
                        }
                    },
                    {"$limit": number_of_docs},
                    {"$addFields": {"score": {"$meta": "searchScore"}}},
                    {
                        "$group": {
                            "_id": "$id",  # Group by `id` field
                            "bestMatch": {
                                "$first": "$$ROOT"
                            },  # Take the highest score (first document after sorting)
                        }
                    },
                    {
                        "$replaceRoot": {
                            "newRoot": "$bestMatch"  # Replace root with the best matching document for each group
                        }
                    },
                    {
                        "$sort": {
                            "score": -1
                        },  # Sort by similarity score in descending order
                    },
                    {"$group": {"_id": None, "docs": {"$push": "$$ROOT"}}},
                    {"$unwind": {"path": "$docs", "includeArrayIndex": "rank"}},
                    {
                        "$addFields": {
                            "fts_score": {
                                "$multiply": [
                                    full_text_weight,
                                    {"$divide": [1.0, {"$add": ["$rank", 1]}]},
                                ]
                            }
                        }
                    },
                    {
                        "$project": {
                            "fts_score": 1,
                            "_id": "$docs._id",
                            "id": "$docs.id",
                            "articleId": "$docs.articleId",
                            "content": "$docs.content",
                            "html_url": "$docs.html_url",
                            "title": "$docs.title",
                        }
                    },
                ],
            }
        },
        {
            "$group": {
                "_id": "$_id",
                "id": {"$first": "$id"},
                "articleId": {"$first": "$articleId"},
                "content": {"$first": "$content"},
                "html_url": {"$first": "$html_url"},
                "title": {"$first": "$title"},
                "vs_score": {"$max": "$vs_score"},
                "fts_score": {"$max": "$fts_score"},
                "cosine_score": {"$max": "$cosine_score"},
            }
        },
        {
            "$group": {
                "_id": "$id",  # Group by `id` field
                "bestMatch": {
                    "$first": "$$ROOT"
                },  # Take the highest score (first document after sorting)
            }
        },
        {
            "$replaceRoot": {
                "newRoot": "$bestMatch"  # Replace root with the best matching document for each group
            }
        },
        {
            "$project": {
                "_id": 1,
                "id": 1,
                "articleId": 1,
                "content": 1,
                "html_url": 1,
                "title": 1,
                "vs_score": {"$ifNull": ["$vs_score", 0]},
                "fts_score": {"$ifNull": ["$fts_score", 0]},
                "cosine_score": {"$ifNull": ["$cosine_score", 0]},
            }
        },
        {
            "$project": {
                "score": {"$add": ["$fts_score", "$vs_score"]},
                "_id": 1,
                "id": 1,
                "vs_score": 1,
                "fts_score": 1,
                "cosine_score": 1,
                "articleId": 1,
                # "content": 1,
                "html_url": 1,
                "title": 1,
            }
        },
        {"$sort": {"score": -1}},
        {"$limit": max_sources},
    ]

    # run pipeline
    result = list(collection.aggregate(pipeline_test))

    return result


def get_new_vector_search(hybrid_query_vector, hybrid_query, max_sources):
    vector_index = "content_embedding_vector_cosine"
    embedding_path = "content_embedding"
    COLLECTION_NAME = "knowledge_base_all"
    collection = db[COLLECTION_NAME]
    number_of_docs = collection.count_documents({})
    pipeline_test = [
        {
            "$vectorSearch": {
                "index": vector_index,
                "path": embedding_path,
                "queryVector": hybrid_query_vector,
                "exact": True,  # exact match ENN # no change
                "limit": number_of_docs,
            }
        },
        {
            "$match": {
                "deleted": False  # Explicitly matching documents where deleted is False
            }
        },
        {
            "$addFields": {
                "cosine_score": {
                    "$subtract": [
                        {"$multiply": [{"$meta": "vectorSearchScore"}, 2]},
                        1,
                    ]
                }
            }
        },
        {"$project": {"content": 0, "content_embedding": 0}},
        {
            "$facet": {
                "vectorResults": [{"$skip": 0}, {"$limit": max_sources}],
                "typeCounts": [
                    {"$group": {"_id": "$type", "vectorCount": {"$sum": 1}}}
                ],
                "typeCountsCategory": [
                    {"$group": {"_id": "$section_id", "vectorCount": {"$sum": 1}}},
                    {
                        "$lookup": {
                            "from": "sections",
                            "localField": "_id",
                            "foreignField": "id",
                            "as": "sectionWithNames",
                        }
                    },
                    {
                        "$addFields": {
                            "sectionName": {
                                "$ifNull": [
                                    {"$arrayElemAt": ["$sectionWithNames.name", 0]},
                                    "Unknown",
                                ]
                            }
                        }
                    },
                    {
                        "$lookup": {
                            "from": "categories",
                            "localField": "sectionWithNames.category_id",
                            "foreignField": "id",
                            "as": "categoryInfo",
                        }
                    },
                    {
                        "$addFields": {
                            "categoryName": {
                                "$ifNull": [
                                    {"$arrayElemAt": ["$categoryInfo.name", 0]},
                                    "No Category",
                                ]
                            }
                        }
                    },
                    {"$match": {"categoryName": {"$ne": "No Category"}}},
                    {
                        "$group": {
                            "_id": "$categoryName",
                            "categoryCount": {"$sum": "$vectorCount"},
                            "sections": {"$push": "$$ROOT"},
                        }
                    },
                    {"$project": {"sectionWithNames": 0, "categoryInfo": 0}},
                ],
                "typeCountsTopic": [
                    {"$group": {"_id": "$topic_id", "vectorCount": {"$sum": 1}}},
                    {
                        "$lookup": {
                            "from": "topics",
                            "localField": "_id",
                            "foreignField": "id",
                            "as": "topicWithNames",
                        }
                    },
                    {
                        "$addFields": {
                            "topicName": {
                                "$ifNull": [
                                    {"$arrayElemAt": ["$topicWithNames.name", 0]},
                                    "No Topic",
                                ]
                            }
                        }
                    },
                    {"$match": {"topicName": {"$ne": "No Topic"}}},
                    {
                        "$group": {
                            "_id": "$topicName",
                            "topicCount": {"$sum": "$vectorCount"},
                            "topics": {"$push": "$$ROOT"},
                        }
                    },
                    {"$project": {"topicWithNames": 0}},
                ],
            }
        },
    ]

    # run pipeline
    result = list(collection.aggregate(pipeline_test))
    return result


def get_label_vector_search(hybrid_query_vector, hybrid_query, max_sources):
    vector_index = "label_names_as_string_embedding_vector_cosine"
    embedding_path = "label_names_as_string_embedding"
    COLLECTION_NAME = "knowledge_base_all"
    collection = db[COLLECTION_NAME]
    number_of_docs = collection.count_documents({})

    main_query = " ".join(hybrid_query.split()[:-2])

    main_query_vector = get_openai_embedding(main_query)

    # Define your match condition for labels (either "netsuite" or both "netsuite" and "primary information")
    label_match_condition = {
        "$and": [
            {
                "label_names": {
                    "$all": [
                        "primary information",  # Matches "primary information" exactly
                        main_query,  # Matches "netsuite" exactly
                    ]
                }
            },  # Articles with both "netsuite" and "primary information"
        ]
    }

    pipeline_test = [
        {
            "$vectorSearch": {
                "index": vector_index,
                "path": embedding_path,
                "queryVector": main_query_vector,
                "exact": True,  # exact match ENN # no change
                "limit": number_of_docs,
            }
        },
        {
            "$match": {
                "deleted": False  # Explicitly matching documents where deleted is False
            }
        },
        {
            "$addFields": {
                "cosine_score": {
                    "$subtract": [
                        {"$multiply": [{"$meta": "vectorSearchScore"}, 2]},
                        1,
                    ]
                }
            }
        },
        # $facet allows us to split into two pipelines
        {
            "$facet": {
                "matched_articles": [
                    # Match articles that satisfy label conditions
                    {"$match": label_match_condition},
                    # Sort the matched articles by article_view_count
                    {"$addFields": {"label_source": "matched_articles"}},
                    {"$sort": {"article_view_count": -1}},
                    # Limit the number of results for matched articles
                    {"$limit": max_sources},
                    {
                        "$project": {
                            "content": 0,
                            "content_embedding": 0,
                            "label_names_as_string_embedding": 0,
                        }
                    },
                ],
                "rest_of_articles": [
                    # Match all articles that do not satisfy label conditions
                    {"$match": {"$nor": [label_match_condition]}},
                    {"$addFields": {"label_source": "rest_of_articles"}},
                    # Sort these articles by search relevance score
                    {"$sort": {"cosine_score": -1}},
                    # Limit the number of results for these articles
                    {"$limit": max_sources},
                    {
                        "$project": {
                            "content": 0,
                            "content_embedding": 0,
                            "label_names_as_string_embedding": 0,
                        }
                    },
                ],
            }
        },
        # Combine both pipelines: matched_articles and rest_of_articles
        {
            "$project": {
                "combined_results": {
                    "$concatArrays": ["$matched_articles", "$rest_of_articles"]
                }
            }
        },
        # Flatten the combined results array
        {"$unwind": "$combined_results"},
        {
            "$project": {
                "content": 0,
                "content_embedding": 0,
                "label_names_as_string_embedding": 0,
            }
        },
        # Limit final result count
        {"$limit": max_sources},
    ]

    # run pipeline
    result = list(collection.aggregate(pipeline_test))
    result = [item["combined_results"] for item in result if "combined_results" in item]
    return result


def get_label_text_search(hybrid_query_vector, hybrid_query, max_sources):
    text_index = "label_names_as_string_index"
    text_param = "label_names"
    COLLECTION_NAME = "knowledge_base_all"
    collection = db[COLLECTION_NAME]

    main_query = " ".join(hybrid_query.split()[:-2])

    # Define your match condition for labels (either "netsuite" or both "netsuite" and "primary information")
    label_match_condition = {
        "$or": [
            {
                "label_names": {
                    "$elemMatch": {
                        "$regex": "^"
                        + main_query
                        + "$",  # Matches "netsuite" with any case
                        "$options": "i",  # Case-insensitive match for "netsuite"
                    }
                }
            },  # Articles with "netsuite" in any case
            {
                "label_names": {
                    "$all": [
                        "primary information",  # Matches "primary information" exactly
                        main_query,  # Matches "netsuite" exactly
                    ]
                }
            },  # Articles with both "netsuite" and "primary information"
        ]
    }

    pipeline_test = [
        # Match stage to filter documents based on label conditions
        {
            "$search": {
                "index": text_index,
                "text": {
                    "query": hybrid_query,  # Add the query text here
                    "path": text_param,
                },
                "returnStoredSource": False,
            }
        },
        {
            "$match": {
                "deleted": False  # Explicitly matching documents where deleted is False
            }
        },
        # Add searchScore as a field in the document
        {
            "$addFields": {
                "searchScore": {
                    "$meta": "searchScore"
                }  # Adds the searchScore to each document
            }
        },
        # $facet allows us to split into two pipelines
        {
            "$facet": {
                "matched_articles": [
                    # Match articles that satisfy label conditions
                    {"$match": label_match_condition},
                    {"$addFields": {"label_source": "matched_articles"}},
                    # Sort the matched articles by article_view_count
                    {"$sort": {"article_view_count": -1}},
                    # Limit the number of results for matched articles
                    {"$limit": max_sources},
                    {
                        "$project": {
                            "content": 0,
                            "content_embedding": 0,
                            "label_names_as_string_embedding": 0,
                        }
                    },
                ],
                "rest_of_articles": [
                    # Match all articles that do not satisfy label conditions
                    {"$match": {"$nor": [label_match_condition]}},
                    {"$addFields": {"label_source": "rest_of_articles"}},
                    # Sort these articles by search relevance score
                    {"$sort": {"searchScore": -1}},
                    # Limit the number of results for these articles
                    {"$limit": max_sources},
                    {
                        "$project": {
                            "content": 0,
                            "content_embedding": 0,
                            "label_names_as_string_embedding": 0,
                        }
                    },
                ],
            }
        },
        # Combine both pipelines: matched_articles and rest_of_articles
        {
            "$project": {
                "combined_results": {
                    "$concatArrays": ["$matched_articles", "$rest_of_articles"]
                }
            }
        },
        # Flatten the combined results array
        {"$unwind": "$combined_results"},
        {
            "$project": {
                "content": 0,
                "content_embedding": 0,
                "label_names_as_string_embedding": 0,
            }
        },
        # Limit final result count
        {"$limit": max_sources},
    ]

    # run pipeline
    result = list(collection.aggregate(pipeline_test))
    result = [item["combined_results"] for item in result if "combined_results" in item]
    return result


def get_boost_search(hybrid_query_vector, hybrid_query, max_sources):
    vector_index = "content_embedding_vector_cosine"
    embedding_path = "content_embedding"
    COLLECTION_NAME = "knowledge_base_all"
    collection = db[COLLECTION_NAME]
    number_of_docs = collection.count_documents({})
    vector_weight = 0.5
    full_text_weight = 0.5
    text_index = "label_names_as_string_index"
    text_param = "label_names"

    pipeline_test = [
        {
            "$vectorSearch": {
                "index": vector_index,
                "path": embedding_path,
                "queryVector": hybrid_query_vector,
                "exact": True,
                "limit": number_of_docs,
            }
        },
        {
            "$match": {
                "deleted": False  # Explicitly matching documents where deleted is False
            }
        },
        {
            "$addFields": {
                "cosine_score": {
                    "$subtract": [
                        {"$multiply": [{"$meta": "vectorSearchScore"}, 2]},
                        1,
                    ]
                }
            }
        },
        {
            "$group": {
                "_id": None,
                "docs": {
                    "$push": {
                        "_id": "$_id",
                        "id": "$id",
                        "cosine_score": "$cosine_score",
                        "title": "$title",
                        "label_names_as_string": "$label_names_as_string",
                        "html_url": "$html_url",
                        "type": "$type",
                    }
                },
            }
        },
        {"$unwind": {"path": "$docs", "includeArrayIndex": "rank"}},
        {
            "$addFields": {
                "vs_score": {
                    "$multiply": [
                        vector_weight,
                        {"$divide": [1.0, {"$add": ["$rank", 1]}]},
                    ]
                }
            }
        },
        {
            "$project": {
                "vs_score": 1,
                "cosine_score": "$docs.cosine_score",
                "_id": "$docs._id",
                "id": "$docs.id",
                "html_url": "$docs.html_url",
                "type": "$docs.type",
                "title": "$docs.title",
                "label_names_as_string": "$docs.label_names_as_string",
            }
        },
        {
            "$unionWith": {
                "coll": COLLECTION_NAME,
                "pipeline": [
                    {
                        "$search": {
                            "index": text_index,
                            "text": {
                                "query": hybrid_query,  # Add the query text here
                                "path": text_param,
                            },
                        }
                    },
                    {
                        "$match": {
                            "deleted": False  # Explicitly matching documents where deleted is False
                        }
                    },
                    {"$limit": number_of_docs},
                    {
                        "$group": {
                            "_id": None,
                            "docs": {
                                "$push": {
                                    "_id": "$_id",
                                    "id": "$id",
                                    "title": "$title",
                                    "html_url": "$html_url",
                                    "type": "$type",
                                }
                            },
                        }
                    },
                    {"$unwind": {"path": "$docs", "includeArrayIndex": "rank"}},
                    {
                        "$addFields": {
                            "fts_score": {
                                "$multiply": [
                                    full_text_weight,
                                    {"$divide": [1.0, {"$add": ["$rank", 1]}]},
                                ]
                            }
                        }
                    },
                    {
                        "$project": {
                            "fts_score": 1,
                            "_id": "$docs._id",
                            "id": "$docs.id",
                            "html_url": "$docs.html_url",
                            "title": "$docs.title",
                            "type": "$docs.type",
                        }
                    },
                ],
            }
        },
        {
            "$group": {
                "_id": "$_id",
                "id": {"$first": "$id"},
                "html_url": {"$first": "$html_url"},
                "title": {"$first": "$title"},
                "label_names_as_string": {"$first": "$label_names_as_string"},
                "type": {"$first": "$type"},
                "vs_score": {"$max": "$vs_score"},
                "fts_score": {"$max": "$fts_score"},
                "cosine_score": {"$max": "$cosine_score"},
            }
        },
        {
            "$group": {
                "_id": "$id",  # Group by `id` field
                "bestMatch": {
                    "$first": "$$ROOT"
                },  # Take the highest score (first document after sorting)
            }
        },
        {
            "$replaceRoot": {
                "newRoot": "$bestMatch"  # Replace root with the best matching document for each group
            }
        },
        {
            "$project": {
                "_id": 1,
                "id": 1,
                "html_url": 1,
                "title": 1,
                "label_names_as_string": 1,
                "type": 1,
                "vs_score": {"$ifNull": ["$vs_score", 0]},
                "fts_score": {"$ifNull": ["$fts_score", 0]},
                "cosine_score": {"$ifNull": ["$cosine_score", 0]},
            }
        },
        {
            "$project": {
                "score": {"$add": ["$fts_score", "$vs_score"]},
                "_id": 1,
                "id": 1,
                "vs_score": 1,
                "fts_score": 1,
                "cosine_score": 1,
                "html_url": 1,
                "title": 1,
                "label_names_as_string": 1,
                "type": 1,
            }
        },
        {"$sort": {"score": -1}},
        {"$limit": max_sources},
    ]

    # run pipeline
    result = list(collection.aggregate(pipeline_test))
    return result


def get_2_vector_search(hybrid_query_vector, hybrid_query, max_sources):
    vector_index = "content_embedding_vector_cosine"
    embedding_path = "content_embedding"
    COLLECTION_NAME = "knowledge_base_all"
    collection = db[COLLECTION_NAME]
    number_of_docs = collection.count_documents({})
    vector_weight = 0.5
    vector_weight_2 = 0.5
    vector_index_2 = "label_names_as_string_embedding_vector_cosine"
    embedding_path_2 = "label_names_as_string_embedding"

    pipeline_test = [
        {
            "$vectorSearch": {
                "index": vector_index,
                "path": embedding_path,
                "queryVector": hybrid_query_vector,
                "exact": True,
                "limit": number_of_docs,
            }
        },
        {
            "$match": {
                "deleted": False  # Explicitly matching documents where deleted is False
            }
        },
        {
            "$addFields": {
                "cosine_score": {
                    "$subtract": [
                        {"$multiply": [{"$meta": "vectorSearchScore"}, 2]},
                        1,
                    ]
                }
            }
        },
        {
            "$group": {
                "_id": None,
                "docs": {
                    "$push": {
                        "_id": "$_id",
                        "id": "$id",
                        "cosine_score": "$cosine_score",
                        "title": "$title",
                        "label_names_as_string": "$label_names_as_string",
                        "html_url": "$html_url",
                        "type": "$type",
                    }
                },
            }
        },
        {"$unwind": {"path": "$docs", "includeArrayIndex": "rank"}},
        {
            "$addFields": {
                "vs_score": {
                    "$multiply": [
                        vector_weight,
                        {"$divide": [1.0, {"$add": ["$rank", 1]}]},
                    ]
                }
            }
        },
        {
            "$project": {
                "vs_score": 1,
                "cosine_score": "$docs.cosine_score",
                "_id": "$docs._id",
                "id": "$docs.id",
                "html_url": "$docs.html_url",
                "type": "$docs.type",
                "title": "$docs.title",
                "label_names_as_string": "$docs.label_names_as_string",
            }
        },
        {
            "$unionWith": {
                "coll": COLLECTION_NAME,
                "pipeline": [
                    {
                        "$vectorSearch": {
                            "index": vector_index_2,
                            "path": embedding_path_2,
                            "queryVector": hybrid_query_vector,
                            "exact": True,
                            "limit": number_of_docs,
                        }
                    },
                    {
                        "$match": {
                            "deleted": False  # Explicitly matching documents where deleted is False
                        }
                    },
                    {
                        "$group": {
                            "_id": None,
                            "docs": {
                                "$push": {
                                    "_id": "$_id",
                                    "id": "$id",
                                    "title": "$title",
                                    "label_names_as_string": "$label_names_as_string",
                                    "html_url": "$html_url",
                                    "type": "$type",
                                }
                            },
                        }
                    },
                    {"$unwind": {"path": "$docs", "includeArrayIndex": "rank"}},
                    {
                        "$addFields": {
                            "fts_score": {
                                "$multiply": [
                                    vector_weight_2,
                                    {"$divide": [1.0, {"$add": ["$rank", 1]}]},
                                ]
                            }
                        }
                    },
                    {
                        "$project": {
                            "fts_score": 1,
                            "_id": "$docs._id",
                            "id": "$docs.id",
                            "html_url": "$docs.html_url",
                            "type": "$docs.type",
                            "title": "$docs.title",
                            "label_names_as_string": "$docs.label_names_as_string",
                        }
                    },
                ],
            }
        },
        {
            "$group": {
                "_id": "$_id",
                "id": {"$first": "$id"},
                "html_url": {"$first": "$html_url"},
                "title": {"$first": "$title"},
                "label_names_as_string": {"$first": "$label_names_as_string"},
                "type": {"$first": "$type"},
                "vs_score": {"$max": "$vs_score"},
                "fts_score": {"$max": "$fts_score"},
                "cosine_score": {"$max": "$cosine_score"},
            }
        },
        {
            "$group": {
                "_id": "$id",  # Group by `id` field
                "bestMatch": {
                    "$first": "$$ROOT"
                },  # Take the highest score (first document after sorting)
            }
        },
        {
            "$replaceRoot": {
                "newRoot": "$bestMatch"  # Replace root with the best matching document for each group
            }
        },
        {
            "$project": {
                "_id": 1,
                "id": 1,
                "html_url": 1,
                "title": 1,
                "label_names_as_string": 1,
                "type": 1,
                "vs_score": {"$ifNull": ["$vs_score", 0]},
                "fts_score": {"$ifNull": ["$fts_score", 0]},
                "cosine_score": {"$ifNull": ["$cosine_score", 0]},
            }
        },
        {
            "$project": {
                "score": {"$add": ["$fts_score", "$vs_score"]},
                "_id": 1,
                "id": 1,
                "vs_score": 1,
                "fts_score": 1,
                "cosine_score": 1,
                "html_url": 1,
                "title": 1,
                "label_names_as_string": 1,
                "type": 1,
            }
        },
        {"$sort": {"score": -1}},
        {"$limit": max_sources},
    ]

    # run pipeline
    result = list(collection.aggregate(pipeline_test))
    return result
