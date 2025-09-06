import streamlit as st
import joblib

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
st.set_page_config(page_title="Fake News Detection", layout="centered")
st.title("📰 Fake News Detection App")
st.write("Paste a news article below and check if it's **Real** or **Fake**.")
user_input = st.text_area("Enter News Article Here:", height=200)
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text first!")
    else:
        # Transform input using TF-IDF vectorizer
        input_tfidf = vectorizer.transform([user_input])
        
        # Make prediction
        prediction = model.predict(input_tfidf)[0]
        label = "✅ Real News" if prediction == 1 else "❌ Fake News"
        
        # Show result
        st.subheader("Prediction:")
        st.success(label)
