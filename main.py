from src.data_preprocessing import preprocess_data
from src.model_training import train_model, save_model

# Define the directory containing the training data
train_dir = 'data/train'

# Preprocess the data
X, y, tfidf = preprocess_data(train_dir)

# Train the model
model = train_model(X, y)

# Save the model and vectorizer
save_model(model, tfidf)
