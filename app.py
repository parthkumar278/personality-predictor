import streamlit as st
import pickle
import re
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text
descriptions = {
    "INTJ": "The Architect — strategic, independent, and always 10 steps ahead.",
    "INTP": "The Thinker — loves ideas, logic, and solving complex puzzles.",
    "INFJ": "The Counselor — deeply empathetic, visionary, and quietly powerful.",
    "INFP": "The Mediator — creative, idealistic, and driven by deep values.",
    "ISTJ": "The Inspector — reliable, detail-oriented, and incredibly dependable.",
    "ISTP": "The Craftsman — calm, practical, and great in a crisis.",
    "ISFJ": "The Protector — warm, caring, and always puts others first.",
    "ISFP": "The Artist — gentle, creative, and lives fully in the moment.",
    "ENTJ": "The Commander — bold, ambitious, and born to lead.",
    "ENTP": "The Debater — clever, curious, and loves a good argument.",
    "ENFJ": "The Teacher — charismatic, inspiring, and uplifts everyone around them.",
    "ENFP": "The Champion — enthusiastic, creative, and endlessly optimistic.",
    "ESTJ": "The Supervisor — organized, assertive, and gets things done.",
    "ESTP": "The Dynamo — energetic, bold, and thrives in the fast lane.",
    "ESFJ": "The Caregiver — sociable, warm, and loves taking care of others.",
    "ESFP": "The Performer — spontaneous, fun-loving, and lights up any room.",
}
@st.cache_resource
def load_or_train():
    df = pd.read_csv("https://raw.githubusercontent.com/parthkumar278/personality-predictor/main/mbti_1.csv")
    df['cleaned'] = df['posts'].apply(clean_text)
    df['IE'] = df['type'].apply(lambda x: 1 if x[0] == 'I' else 0)
    df['NS'] = df['type'].apply(lambda x: 1 if x[1] == 'N' else 0)
    df['TF'] = df['type'].apply(lambda x: 1 if x[2] == 'T' else 0)
    df['JP'] = df['type'].apply(lambda x: 1 if x[3] == 'J' else 0)
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
    X = tfidf.fit_transform(df['cleaned'])
    models = {}
    for dim in ['IE', 'NS', 'TF', 'JP']:
        model = LogisticRegression(max_iter=1000)
        model.fit(X, df[dim])
        models[dim] = model
    return models, tfidf

st.set_page_config(page_title="MBTI Personality Predictor", page_icon="🧠")
st.title("🧠 MBTI Personality Predictor")
st.write("Write anything about yourself and we'll predict your personality type!")

with st.spinner("Loading model... (first load takes ~1 min)"):
    models, tfidf = load_or_train()

user_input = st.text_area("Your text here...", height=150)

if st.button("Predict my personality"):
    if len(user_input.strip()) == 0:
        st.warning("Please write something first!")
    else:
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])
        ie = "I" if models['IE'].predict(vector)[0] == 1 else "E"
        ns = "N" if models['NS'].predict(vector)[0] == 1 else "S"
        tf = "T" if models['TF'].predict(vector)[0] == 1 else "F"
        jp = "J" if models['JP'].predict(vector)[0] == 1 else "P"
        mbti = ie + ns + tf + jp
        st.success(f"Your predicted personality type is: **{mbti}**")
        st.info(descriptions[mbti])
        st.subheader("Dimension Breakdown")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("I/E", ie, f"{models['IE'].predict_proba(vector).max():.0%} confident")
            st.metric("N/S", ns, f"{models['NS'].predict_proba(vector).max():.0%} confident")
        with col2:
            st.metric("T/F", tf, f"{models['TF'].predict_proba(vector).max():.0%} confident")
            st.metric("J/P", jp, f"{models['JP'].predict_proba(vector).max():.0%} confident")