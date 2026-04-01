import streamlit as st
import pickle
import re

# Load saved model and vectorizer
models = pickle.load(open("models.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# Clean text function
def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text

# Page config
st.set_page_config(page_title="MBTI Personality Predictor", page_icon="🧠")

# Title
st.title("🧠 MBTI Personality Predictor")
st.write("Write anything about yourself — your thoughts, opinions, or how you feel — and we'll predict your personality type!")

# Text input
user_input = st.text_area("Your text here...", height=150)

# Predict button
if st.button("Predict my personality"):
    if len(user_input.strip()) == 0:
        st.warning("Please write something first!")
    else:
        # Clean and vectorize
        cleaned = clean_text(user_input)
        vector = tfidf.transform([cleaned])

        # Predict each dimension
        ie = "I" if models['IE'].predict(vector)[0] == 1 else "E"
        ns = "N" if models['NS'].predict(vector)[0] == 1 else "S"
        tf = "T" if models['TF'].predict(vector)[0] == 1 else "F"
        jp = "J" if models['JP'].predict(vector)[0] == 1 else "P"

        mbti = ie + ns + tf + jp

        # Confidence scores
        ie_conf = models['IE'].predict_proba(vector).max()
        ns_conf = models['NS'].predict_proba(vector).max()
        tf_conf = models['TF'].predict_proba(vector).max()
        jp_conf = models['JP'].predict_proba(vector).max()

        # Show result
        st.success(f"Your predicted personality type is: **{mbti}**")

        # Show dimension breakdown
        st.subheader("Dimension Breakdown")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("I/E", ie, f"{ie_conf:.0%} confident")
            st.metric("N/S", ns, f"{ns_conf:.0%} confident")

        with col2:
            st.metric("T/F", tf, f"{tf_conf:.0%} confident")
            st.metric("J/P", jp, f"{jp_conf:.0%} confident")