import streamlit as st
import pickle

# Load the trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Page title
st.title("📱 Sentiment Analysis of Smartphone Reviews")

# Instruction
st.markdown("Enter a product review in the box below. Click the button to know if it's Positive, Negative, or Neutral.")

# Text input
review = st.text_area("✍️ Type your review here:")

# Predict on button click
if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review first.")
    else:
        # Vectorize the text
        vectorized_review = vectorizer.transform([review])
        
        # Predict sentiment
        prediction = model.predict(vectorized_review)[0]

        # Map numeric prediction to labels
        if prediction == 2:
            st.success("✅ Positive Sentiment")
        elif prediction == 1:
            st.info("😐 Neutral Sentiment")
        elif prediction == 0:
            st.error("❌ Negative Sentiment")
        else:
            st.warning("🤔 Unexpected result")
