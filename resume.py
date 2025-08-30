# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 19:11:43 2025

@author: Bharath
"""

import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text


@st.cache_data
def load_data():
    df = pd.read_csv("Resume.csv")
    df['cleaned_resume'] = df['Resume'].apply(clean_text)
    return df

df = load_data()


st.title("🤖 AI-Powered Resume Screening System")
st.write("Upload a job description and get top matching resumes from dataset")


job_description = st.text_area("Enter Job Description:")

if st.button("FindTop Candidates"):
    if job_description.strip() == "":
        st.warning("Please enter a job description")
    else:
        job_description = clean_text(job_description)

       
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(df['cleaned_resume'].tolist() + [job_description])

        similarity_scores = cosine_similarity(tfidf_matrix[:-1], tfidf_matrix[-1])

    
        df['match_score'] = similarity_scores

       
        top_candidates = df[['Category', 'Resume', 'match_score']].sort_values(by="match_score", ascending=False).head(10)

        st.subheader("📌 Top Candidates")
        for i, row in top_candidates.iterrows():
            st.markdown(f"**{row['Category']}** (Score: {row['match_score']:.2f})")
            st.write(row['Resume'][:400] + "...")
            st.write("---")