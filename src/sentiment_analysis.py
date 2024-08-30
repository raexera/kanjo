import os
import joblib

def load_model(model_path='models/sentiment_model.pkl', vectorizer_path='models/tfidf_vectorizer.pkl'):
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        return model, vectorizer
    else:
        return None, None

def predict_sentiment(model, vectorizer, text):
    # Preprocess the input text and make a prediction
    processed_text = vectorizer.transform([text])
    prediction = model.predict(processed_text)[0]
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    return sentiment
