import streamlit as st
import joblib
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download NLTK data (only once)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load models and vectorizer
st.set_page_config(page_title="Toxic Comment Classifier")
st.title("🛡️ Toxic Comment Classifier")
st.write("This app detects multiple types of toxicity in user comments.")

# Load vectorizer and models
tfidf = joblib.load("tfidf_vectorizer.pkl")
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
models = {label: joblib.load(f"toxic_model_{label}.pkl") for label in labels}

# Text cleaner
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#','', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Prediction logic
def predict(comment):
    cleaned = clean_text(comment)
    vector = tfidf.transform([cleaned])
    results = {label: "✅ Yes" if models[label].predict(vector)[0] == 1 else "❌ No" for label in labels}
    return results

# UI Input
user_input = st.text_area("Enter a comment to classify:")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a comment.")
    else:
        with st.spinner("Analyzing..."):
            output = predict(user_input)
            st.subheader("Prediction Results:")
            for label, result in output.items():
                st.write(f"**{label.replace('_', ' ').capitalize()}**: {result}")
