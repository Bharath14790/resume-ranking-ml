# streamlit_app.py
import streamlit as st
import pickle
import re
from sklearn.metrics.pairwise import cosine_similarity

# Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Load Pickle model
@st.cache_resource
def load_model():
    with open("resume_model.pkl", "rb") as f:
        df, vectorizer, tfidf_matrix = pickle.load(f)
    return df, vectorizer, tfidf_matrix

df, vectorizer, tfidf_matrix = load_model()

st.title("🤖 AI-Powered Resume Screening System")
st.write("Upload a job description and get top matching resumes from dataset")

job_description = st.text_area("Enter Job Description:")

if st.button("Find Top Candidates"):
    if job_description.strip() == "":
        st.warning("Please enter a job description")
    else:
        job_description = clean_text(job_description)

        # Transform job description
        job_vec = vectorizer.transform([job_description])

        # Similarity
        similarity_scores = cosine_similarity(tfidf_matrix, job_vec).flatten()

        df['match_score'] = similarity_scores

        # Top 10 candidates
        top_candidates = df[['Category', 'Resume', 'match_score']].sort_values(
            by="match_score", ascending=False
        ).head(10)

        st.subheader("📌 Top Candidates")
        for i, row in top_candidates.iterrows():
            st.markdown(f"**{row['Category']}** (Score: {row['match_score']:.2f})")
            st.write(row['Resume'][:400] + "...")
            st.write("---")
