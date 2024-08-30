from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_model(X, y, model_type='Naive Bayes'):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if model_type == 'Naive Bayes':
        model = MultinomialNB()
    elif model_type == 'SVM':
        model = SVC(probability=True)
    elif model_type == 'Random Forest':
        model = RandomForestClassifier()
    elif model_type == 'Ensemble':
        model = VotingClassifier(estimators=[
            ('nb', MultinomialNB()),
            ('svm', SVC(probability=True)),
            ('rf', RandomForestClassifier())
        ], voting='soft')
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')
    
    return model

def save_model(model, tfidf, model_path='models/sentiment_model.pkl', vectorizer_path='models/tfidf_vectorizer.pkl'):
    # Save the model and vectorizer
    joblib.dump(model, model_path)
    joblib.dump(tfidf, vectorizer_path)
