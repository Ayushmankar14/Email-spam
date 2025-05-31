import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)

    filtered_tokens = [token for token in tokens if token.isalnum()]
    filtered_tokens = [token for token in filtered_tokens if token not in stopwords.words('english')]
    stemmed_tokens = [ps.stem(token) for token in filtered_tokens]

    return " ".join(stemmed_tokens)


@st.cache_resource
def load_model_vectorizer():
    try:
        vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
        clf_model = pickle.load(open('model.pkl', 'rb'))
        return vectorizer, clf_model
    except Exception as e:
        st.error(f"Error loading model/vectorizer: {e}")
        return None, None


tfidf, model = load_model_vectorizer()

st.title("📧 Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if not input_sms.strip():
        st.warning("Please enter a message before clicking Predict.")
    elif tfidf is None or model is None:
        st.error("Model or vectorizer not loaded properly. Please check your files.")
    else:
        # Preprocess input
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])

        # Convert to dense if model requires (e.g., SVC trained on dense)
        try:
            prediction = model.predict(vector_input.toarray())[0]
        except AttributeError:
            # If model accepts sparse input (e.g., MultinomialNB), fallback:
            prediction = model.predict(vector_input)[0]

        if prediction == 1:
            st.header("🚫 Spam")
        else:
            st.header("✅ Not Spam")
