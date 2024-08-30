import os
import glob
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


def load_reviews_from_directory(directory, label):
    reviews = []
    for file_path in glob.glob(os.path.join(directory, "*.txt")):
        with open(file_path, "r", encoding="utf-8") as file:
            review_text = file.read()
            reviews.append({"review": review_text, "sentiment": label})
    return reviews


def preprocess_data(train_dir=None, custom_dataset=None):
    if custom_dataset is not None:
        data = pd.read_csv(custom_dataset)
        data["sentiment"] = data["sentiment"].map({"pos": 1, "neg": 0})
    else:
        pos_reviews = load_reviews_from_directory(os.path.join(train_dir, "pos"), 1)
        neg_reviews = load_reviews_from_directory(os.path.join(train_dir, "neg"), 0)
        data = (
            pd.DataFrame(pos_reviews + neg_reviews)
            .sample(frac=1)
            .reset_index(drop=True)
        )

    # Tokenization and Stopword removal
    data["processed_text"] = data["review"].apply(
        lambda x: " ".join(word for word in x.split() if word.lower() not in stop_words)
    )

    # Vectorization using TF-IDF
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(data["processed_text"])

    # Labels
    y = data["sentiment"]

    return X, y, tfidf
