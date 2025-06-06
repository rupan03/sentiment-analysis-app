import streamlit as st
import pickle
import re
import string
import nltk

# If not already done once, uncomment this line and run once
# nltk.download('stopwords')

from nltk.corpus import stopwords

# Define text preprocessing (same as your notebook!)
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()  # Trim whitespace
    stop_words = set(stopwords.words('english'))  # Load stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# App title and styling
st.set_page_config(page_title="📱 Smartphone Review Sentiment Analyzer", page_icon="📱")
st.title("📱 Smartphone Review Sentiment Analyzer")
st.markdown("### Analyze if your review is Positive, Neutral, or Negative 🎯")
st.markdown("---")

# Text input
review = st.text_area("✍️ Enter your smartphone review here:", height=150)

# Analyze button
if st.button("🚀 Analyze Sentiment"):
    if review.strip() == "":
        st.warning("⚠️ Please enter a review first.")
    else:
        with st.spinner('Analyzing... please wait ⏳'):
            # Preprocess the input
            cleaned_review = preprocess_text(review)
            
            # Vectorize
            vectorized_review = vectorizer.transform([cleaned_review])
            
            # Predict
            prediction = model.predict(vectorized_review)[0]
            
            # Map prediction to label
            label_mapping = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            human_label = label_mapping.get(prediction, "🤔 Unexpected result")
        
        # Display result
        st.markdown("---")
        st.subheader("🎉 Result:")

        if human_label == "Positive":
            st.success("✅ Positive Sentiment 😊")
            st.balloons()
        elif human_label == "Neutral":
            st.info("😐 Neutral Sentiment 😐")
        elif human_label == "Negative":
            st.error("❌ Negative Sentiment 😞")
        else:
            st.warning(human_label)

# Footer
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit.")
