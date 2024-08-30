import streamlit as st
from src.data_preprocessing import preprocess_data
from src.model_training import train_model, save_model
from src.sentiment_analysis import load_model, predict_sentiment
import pandas as pd

# App title
st.title("AI-Powered Sentiment Analysis Tool")

# Model selection
model_type = st.selectbox("Select Model", ("Naive Bayes", "SVM", "Random Forest", "Ensemble"))

# Custom dataset upload
uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type="csv")

# Load model and vectorizer if not training a new model
model, vectorizer = None, None
if not uploaded_file:
    model, vectorizer = load_model()
    if model is None or vectorizer is None:
        st.write("No pre-trained model found. Please upload a custom dataset to train a new model.")

# User input for sentiment analysis
user_input = st.text_area("Enter text for sentiment analysis:")

# Analyze sentiment
if st.button("Analyze"):
    if user_input:
        if uploaded_file:
            # Train with custom dataset
            X, y, tfidf = preprocess_data(custom_dataset=uploaded_file)
            model = train_model(X, y, model_type=model_type)
            save_model(model, tfidf)
            vectorizer = tfidf  # Assign vectorizer after training
        elif model is not None and vectorizer is not None:
            # Use pre-trained model
            sentiment = predict_sentiment(model, vectorizer, user_input)
        else:
            st.write("No model available for prediction. Please upload a dataset to train a model.")
            st.stop()  # Stop further execution

        if model is not None and vectorizer is not None:
            # Prediction details
            processed_text = vectorizer.transform([user_input])
            probability = model.predict_proba(processed_text)[0]
            prediction = model.predict(processed_text)[0]
            sentiment = 'Positive' if prediction == 1 else 'Negative'
            confidence = "High" if max(probability) > 0.75 else "Medium" if max(probability) > 0.5 else "Low"

            # Display result
            st.write(f"Sentiment: **{sentiment}** (Confidence: {confidence})")

            # History and Visualization
            if 'history' not in st.session_state:
                st.session_state.history = []

            st.session_state.history.append({
                'Model': model_type,
                'Text': user_input,
                'Prediction': sentiment,
                'Probability': f"{max(probability):.2f}",
                'Confidence': confidence
            })
            history_df = pd.DataFrame(st.session_state.history)

            st.write("### Analysis History")
            st.dataframe(history_df)

            st.write("### Sentiment Distribution")
            sentiment_distribution = history_df['Prediction'].value_counts()
            st.bar_chart(sentiment_distribution)

            st.write("### Prediction Confidence Levels")
            confidence_levels = history_df['Confidence'].value_counts()
            st.bar_chart(confidence_levels)

            st.write("### Prediction Probability Scores")
            probability_scores = history_df['Probability'].astype(float)
            st.bar_chart(pd.DataFrame({'Probability': probability_scores}))

    else:
        st.write("Please enter some text.")

# Display Model Performance if model was trained with a custom dataset
if uploaded_file and model is not None:
    st.write("### Model Performance")
    accuracy = model.score(X, y)
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")
