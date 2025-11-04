# sentiment_analyzer.py
# AI-Driven Sentiment Analyzer - Model Training

import re
import nltk
import joblib
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

print("📥 Loading IMDB dataset...")
dataset = load_dataset("imdb")
train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])
df = pd.concat([train_df, test_df]).reset_index(drop=True)
print("✅ Dataset loaded successfully!")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

print("🧹 Cleaning text...")
df["clean_text"] = df["text"].apply(clean_text)

X = df["clean_text"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("🔠 Converting to TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("🧠 Training Logistic Regression...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)
print("✅ Accuracy:", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

joblib.dump(model, "sentiment_model.joblib")
joblib.dump(vectorizer, "tfidf_vectorizer.joblib")
print("💾 Model and vectorizer saved successfully!")

def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()
    sentiment = "🙂 Positive" if pred == 1 else "☹️ Negative"
    print(f"\nInput: {text}")
    print(f"Prediction: {sentiment} (Confidence: {round(prob*100,2)}%)")

# Quick test
predict_sentiment("I loved this movie, it was fantastic!")
predict_sentiment("The movie was boring and too long.")

