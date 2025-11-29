import streamlit as st
from retrieval import retrieve_faiss, retrieve_faiss_image, retrieve_tfidf, retrieve_bm25
import os
from io import BytesIO
import sys

# Get project root directory (one level up from app/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)
from Paths import video_path as video_path_config

# Page setup
st.set_page_config(page_title="Video QA System", page_icon="🎥", layout="centered")

st.title("🎥 Video Retrieval-Augmented QA System")
st.markdown("""
<style>
/* Constrain main content width for screenshots */
.main .block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.question-block {
    background-color: #ffffff;
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 40px;
    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s;
}
.question-block:hover {
    transform: scale(1.02);
}
.qa-header {
    font-size: 24px;
    font-weight: bold;
    color: #222;
    margin-bottom: 15px;
}
.qa-label {
    font-size: 20px;
    color: #444;
    margin-top: 10px;
}
.qa-timestamp {
    font-size: 18px;
    color: #1f77b4;
    margin-bottom: 20px;
}
.stVideo {
    margin-bottom: 15px;
}
.section-header {
    font-size: 22px;
    font-weight: bold;
    color: #1f77b4;
    margin-top: 20px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# Video Display
video_path = os.path.join(project_root, video_path_config)
if os.path.exists(video_path):
    st.video(video_path, start_time=0)
else:
    st.error("Video not found!")
if "qa_pairs" not in st.session_state:
    st.session_state.qa_pairs = []


query = st.text_input("🔎 Enter your question here:", placeholder="Type your question about the video...")
uploaded_image = st.file_uploader("🖼️ Optional: Upload an image for visual query", type=["png", "jpg", "jpeg"])

if st.button("Ask Question", use_container_width=True):
    if query.strip() == "":
        st.warning("Please enter a valid question!")
    else:
        faiss_timestamp = retrieve_faiss(query)
        tfidf_timestamp = retrieve_tfidf(query)
        bm25_timestamp = retrieve_bm25(query)
        image_timestamp = None
        if uploaded_image is not None:
            image_bytes = uploaded_image.read()
            image_timestamp = retrieve_faiss_image(BytesIO(image_bytes))

        # Save Q&A Pair
        st.session_state.qa_pairs.append({
            "question": query,
            "faiss_timestamp": faiss_timestamp,
            "tfidf_timestamp": tfidf_timestamp,
            "bm25_timestamp": bm25_timestamp,
            "image_timestamp": image_timestamp,
            "image_data": uploaded_image.getvalue() if uploaded_image else None,
            "image_name": uploaded_image.name if uploaded_image else None
        })

st.subheader("💡 Previous Questions and Answers")
for idx, pair in enumerate(st.session_state.qa_pairs[::-1]):
    with st.container():
        st.markdown("<div class='question-block'>", unsafe_allow_html=True)
        num = len(st.session_state.qa_pairs) - idx
        st.markdown(f"<div class='qa-header'>💬 Q{num}: {pair['question']}</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-header'>🔵 Semantic Search (FAISS Text)</div>", unsafe_allow_html=True)
        if pair['faiss_timestamp'] is not None:
            st.video(video_path, start_time=int(pair['faiss_timestamp']))
            st.markdown(f"<div class='qa-timestamp'>Starts at <strong>{int(pair['faiss_timestamp'])}</strong> seconds (FAISS Text).</div>", unsafe_allow_html=True)
        else:
            st.error("No FAISS text result found.")

        st.markdown("<div class='section-header'>🟣 Visual Search (FAISS Image)</div>", unsafe_allow_html=True)
        if pair['image_data']:
            st.image(pair['image_data'], caption=pair['image_name'], use_container_width=False)
            if pair['image_timestamp'] is not None:
                st.video(video_path, start_time=int(pair['image_timestamp']))
                st.markdown(f"<div class='qa-timestamp'>Starts at <strong>{int(pair['image_timestamp'])}</strong> seconds (FAISS Image).</div>", unsafe_allow_html=True)
            else:
                st.error("No FAISS image result found.")
        else:
            st.info("No image query provided.")

        st.markdown("<div class='section-header'>🟢 Lexical Search (TF-IDF & BM25)</div>", unsafe_allow_html=True)
        cols = st.columns(2)
        with cols[0]:
            st.markdown("<div class='qa-label'>🔹 TF-IDF Result:</div>", unsafe_allow_html=True)
            if pair['tfidf_timestamp'] is not None:
                st.video(video_path, start_time=int(pair['tfidf_timestamp']))
                st.markdown(f"<div class='qa-timestamp'>Starts at <strong>{int(pair['tfidf_timestamp'])}</strong> seconds (TF-IDF).</div>", unsafe_allow_html=True)
            else:
                st.error("No TF-IDF result found.")
        with cols[1]:
            st.markdown("<div class='qa-label'>🔸 BM25 Result:</div>", unsafe_allow_html=True)
            if pair['bm25_timestamp'] is not None:
                st.video(video_path, start_time=int(pair['bm25_timestamp']))
                st.markdown(f"<div class='qa-timestamp'>Starts at <strong>{int(pair['bm25_timestamp'])}</strong> seconds (BM25).</div>", unsafe_allow_html=True)
            else:
                st.error("No BM25 result found.")

        st.markdown("</div>", unsafe_allow_html=True)
