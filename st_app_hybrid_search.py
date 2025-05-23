import time

import streamlit as st

st.set_page_config(
    page_title="Mongo Hybrid Search Demo",
    page_icon=":mag:",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Mongo Hybrid Search Demo")

# Apply custom CSS to adjust title size
st.markdown(
    """
    <style>
        .custom-title {
            font-size: 20px !important;  /* Adjust this size as needed */
            font-weight: bold;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# Apply custom CSS to adjust title size
st.markdown(
    """
    <style>
        .custom-title-2 {
            font-size: 20px !important;  /* Adjust this size as needed */
        }
    </style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f'<p class="custom-title-2">Hybrid Search - Vector Weight 0.5 , Text Based Search 0.5</p>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<p class="custom-title-2">Hybrid Search - Chunking - Vector Weight 0.75 , Text Based Search 0.25</p>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<p class="custom-title-2">Hybrid Search - 2 Vector - Content Vector Weight 0.5 , Label Vector Weight 0.5</p>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<p class="custom-title-2">For Content Vector Search Latest - Time Taken Includes count computation too</p>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<p class="custom-title-2">For Last Four Search Type - Articles and Community Data till 22 Jan 2025</p>',
    unsafe_allow_html=True,
)

st.markdown(
    f'<p class="custom-title-2">"Label-Promoted - View-Sorted - Relevance Search" is a MongoDB query that finds articles with specific labels (like "primary information" and the search query), sorts them by view count (till 7 Feb) for popularity, and then performs a vector search to rank other relevant articles. Optimised for 1 word or 2 word query</p>',
    unsafe_allow_html=True,
)

from kb_query_types import (
    get_2_vector_search,
    get_boost_search,
    get_chunking_search,
    get_hybrid_chunking_search,
    get_hybrid_search,
    get_label_text_search,
    get_label_vector_search,
    get_new_vector_search,
    get_openai_embedding,
    get_vector_search,
)

search_methods = [
    ("Vector", get_vector_search),
    (
        "Hybrid",
        get_hybrid_search,
    ),
    ("Vector - Chunk", get_chunking_search),
    (
        " Hybrid - Chunk",
        get_hybrid_chunking_search,
    ),
    ("Content Vector Latest", get_new_vector_search),
    ("Label - Promoted & Sort ", get_label_vector_search),
    ("Label Text - Order - Views", get_label_text_search),
    ("Ct Vec + Lbl Text - Hybrid", get_boost_search),
    ("Ct Vec + Lbl Vec - Hybrid", get_2_vector_search),
]


def main():
    user_input = st.text_input("Search", value="netsuite")
    words = user_input.split()
    if len(words) == 1 or len(words) == 2:
        user_input = user_input.lower() + " primary information"
        call_all_methods = True
    else:
        call_all_methods = False
    hybrid_query_vector = get_openai_embedding(user_input)
    submit = st.button("Search")
    # (c1, c2, c3, c4, c5, c6, c7, c8, c9) = st.columns([1, 1, 1, 1, 1, 1, 1, 1, 1])
    (c2, c5, c6, c8, c9) = st.columns([1, 1, 1, 1, 1])
    if submit:
        # c1.markdown(
        #     f'<p class="custom-title">{search_methods[0][0]}</p>',
        #     unsafe_allow_html=True,
        # )
        # c1.markdown(
        #     f'<p class="custom-title"></p>',
        #     unsafe_allow_html=True,
        # )
        # c1.markdown(
        #     f'<p class="custom-title"></p>',
        #     unsafe_allow_html=True,
        # )
        # t1 = time.time()
        # results = search_methods[0][1](hybrid_query_vector, user_input, 10)
        # t2 = time.time()
        # c1.write(f"Time taken {round(t2-t1, 2)}s")
        # c1.markdown("---")
        # for result in results[:10]:
        #     # c1.write(result["title"])
        #     # c1.write(result["html_url"])
        #     # c1.markdown(result["title"] + " [link](%s)" % result["html_url"])
        #     link_html = "[{}]({})".format(result["title"], result["html_url"])
        #     c1.markdown(link_html)
        #     # c1.markdown("[ {} ] ( {} )".format(result["title"], result["html_url"]))
        #     # c1.write("vector cosine score: " + str(result["cosine_score"]))
        #     c1.markdown("---")
        c2.markdown(
            f'<p class="custom-title">{search_methods[1][0]}</p>',
            unsafe_allow_html=True,
        )
        t1 = time.time()
        results = search_methods[1][1](hybrid_query_vector, user_input, 10)
        t2 = time.time()
        c2.write(f"Time taken {round(t2-t1, 2)}s")
        c2.markdown("---")
        for result in results[:10]:
            # c2.write(result["title"])
            # c2.write(result["html_url"])
            # c2.markdown(result["title"] + " [link](%s)" % result["html_url"])
            link_html = "[{}]({})".format(result["title"], result["html_url"])
            c2.markdown(link_html)
            # c2.write("vector cosine score: " + str(result["cosine_score"]))
            # c2.write("vector rank score: " + str(result["vs_score"]))
            # c2.write("text based rank score: " + str(result["fts_score"]))
            # c2.write("total rank score: " + str(result["score"]))
            c2.markdown("---")
        # c3.markdown(
        #     f'<p class="custom-title">{search_methods[2][0]}</p>',
        #     unsafe_allow_html=True,
        # )
        # c3.markdown(
        #     f'<p class="custom-title"></p>',
        #     unsafe_allow_html=True,
        # )
        # c3.markdown(
        #     f'<p class="custom-title"></p>',
        #     unsafe_allow_html=True,
        # )
        # t1 = time.time()
        # results = search_methods[2][1](hybrid_query_vector, user_input, 10)
        # t2 = time.time()
        # c3.write(f"Time taken {round(t2-t1, 2)}s")
        # c3.markdown("---")
        # for result in results[:10]:
        #     # c3.write(result["title"])
        #     # c3.write(result["html_url"])
        #     # c3.markdown(result["title"] + " [link](%s)" % result["html_url"])
        #     link_html = "[{}]({})".format(result["title"], result["html_url"])
        #     c3.markdown(link_html)
        #     # c3.markdown("check out this [link](%s)" % result["html_url"])
        #     # c3.write("vector cosine score: " + str((2 * result["score"]) - 1))
        #     c3.markdown("---")
        # c4.markdown(
        #     f'<p class="custom-title">{search_methods[3][0]}</p>',
        #     unsafe_allow_html=True,
        # )
        # c4.markdown(
        #     f'<p class="custom-title"></p>',
        #     unsafe_allow_html=True,
        # )
        # c4.markdown(
        #     f'<p class="custom-title"></p>',
        #     unsafe_allow_html=True,
        # )
        # t1 = time.time()
        # results = search_methods[3][1](hybrid_query_vector, user_input, 10)
        # t2 = time.time()
        # c4.write(f"Time taken {round(t2-t1, 2)}s")
        # c4.markdown("---")
        # for result in results[:10]:
        #     # c4.write(result["title"])
        #     # c4.markdown(result["title"] + " [link](%s)" % result["html_url"])
        #     link_html = "[{}]({})".format(result["title"], result["html_url"])
        #     c4.markdown(link_html)
        #     # c4.write("vector cosine score: " + str(result["cosine_score"]))
        #     # c4.write("vector rank score: " + str(result["vs_score"]))
        #     # c4.write("text based rank score: " + str(result["fts_score"]))
        #     # c4.write("total rank score: " + str(result["score"]))
        #     c4.markdown("---")
        c5.markdown(
            f'<p class="custom-title">{search_methods[4][0]}</p>',
            unsafe_allow_html=True,
        )
        t1 = time.time()
        results = search_methods[4][1](hybrid_query_vector, user_input, 10)
        t2 = time.time()
        c5.write(f"Time taken {round(t2-t1, 2)}s")
        c5.markdown("---")
        for result in results[0]["vectorResults"]:
            link_html = "[{}]({})".format(result["title"], result["html_url"])
            c5.markdown(link_html)
            c5.write("Type: " + str(result["type"]))
            c5.markdown("---")
        st.sidebar.markdown(" ### Article and Community Type Counts")
        for result in results[0]["typeCounts"]:
            st.sidebar.write(result["_id"] + " : " + str(result["vectorCount"]))
        st.sidebar.markdown("---")
        st.sidebar.markdown(" ### Category Types Counts")
        for result in results[0]["typeCountsCategory"]:
            st.sidebar.write(result["_id"] + " : " + str(result["categoryCount"]))
        st.sidebar.markdown("---")
        st.sidebar.markdown(" ### Topic Types Counts")
        for result in results[0]["typeCountsTopic"]:
            st.sidebar.write(str(result["_id"]) + " : " + str(result["topicCount"]))
        st.sidebar.markdown("---")
        c6.markdown(
            f'<p class="custom-title">{search_methods[5][0]}</p>',
            unsafe_allow_html=True,
        )
        t1 = time.time()
        results = search_methods[5][1](hybrid_query_vector, user_input, 10)
        t2 = time.time()
        c6.write(f"Time taken {round(t2-t1, 2)}s")
        c6.markdown("---")
        if call_all_methods == False:
            c6.write("This search will not be used for more than 2 word query")
        else:
            for result in results[:10]:
                link_html = "[{}]({})".format(result["title"], result["html_url"])
                c6.markdown(link_html)
                # c6.write("Source: " + str(result["label_source"]))
                if "article_view_count" in result:
                    c6.write("Article View Count: " + str(result["article_view_count"]))
                else:
                    c6.write("Article View Count: -")
                c6.write("Type: " + str(result["type"]))
                c6.write("Label Names: " + str(result["label_names_as_string"]))
                c6.write("vector cosine score: " + str(result["cosine_score"]))
                c6.markdown("---")
        # c7.markdown(
        #     f'<p class="custom-title">{search_methods[6][0]}</p>',
        #     unsafe_allow_html=True,
        # )
        # t1 = time.time()
        # results = search_methods[6][1](hybrid_query_vector, user_input, 10)
        # t2 = time.time()
        # c7.write(f"Time taken {round(t2-t1, 2)}s")
        # c7.markdown("---")
        # for result in results[:10]:
        #     link_html = "[{}]({})".format(result["title"], result["html_url"])
        #     c7.markdown(link_html)
        #     # c7.write("score: " + str(result["score"]))
        #     # c7.write("Source: " + str(result["label_source"]))
        #     c7.write("Article View Count: " + str(result["article_view_count"]))
        #     c7.write("Type: " + str(result["type"]))
        #     c7.write("Label Names: " + str(result["label_names_as_string"]))
        #     c7.markdown("---")
        c8.markdown(
            f'<p class="custom-title">{search_methods[7][0]}</p>',
            unsafe_allow_html=True,
        )
        t1 = time.time()
        results = search_methods[7][1](hybrid_query_vector, user_input, 10)
        t2 = time.time()
        c8.write(f"Time taken {round(t2-t1, 2)}s")
        c8.markdown("---")
        if call_all_methods == False:
            c8.write("This search will not be used for more than 2 word query")
        else:
            for result in results[:10]:
                link_html = "[{}]({})".format(result["title"], result["html_url"])
                c8.markdown(link_html)
                c8.write("Type: " + str(result["type"]))
                if "label_names_as_string" in result:
                    c8.write("Label Names: " + str(result["label_names_as_string"]))
                c8.markdown("---")
        c9.markdown(
            f'<p class="custom-title">{search_methods[8][0]}</p>',
            unsafe_allow_html=True,
        )
        t1 = time.time()
        results = search_methods[8][1](hybrid_query_vector, user_input, 10)
        t2 = time.time()
        c9.write(f"Time taken {round(t2-t1, 2)}s")
        c9.markdown("---")
        if call_all_methods == False:
            c9.write("This search will not be used for more than 2 word query")
        else:
            for result in results[:10]:
                link_html = "[{}]({})".format(result["title"], result["html_url"])
                c9.markdown(link_html)
                c9.write("Type: " + str(result["type"]))
                if "label_names_as_string" in result:
                    c9.write("Label Names: " + str(result["label_names_as_string"]))
                c9.markdown("---")


if __name__ == "__main__":
    main()
