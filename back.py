import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Load Dataset
#C:\Users\rejas\Prakash Senapati sir\Prakash Senapati lab\4.CUSTOMERS REVIEW DATASET\Restaurant_Reviews.tsv
def load_dataset(filepath, delimiter='\t'):
    return pd.read_csv(filepath, delimiter=delimiter, quoting=3)

# Clean Text Data
def clean_texts(data, column, remove_stopwords=True):
    nltk.download('stopwords')
    corpus = []
    ps = PorterStemmer()
    for text in data[column]:
        review = re.sub('[^a-zA-Z]', ' ', text)
        review = review.lower()
        review = review.split()
        if remove_stopwords:
            review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
        else:
            review = [ps.stem(word) for word in review]
        corpus.append(' '.join(review))
    return corpus

# Feature Extraction
def vectorize_texts(corpus, method="bow"):
    if method == "bow":
        vectorizer = CountVectorizer()
    elif method == "tfidf":
        vectorizer = TfidfVectorizer()
    else:
        raise ValueError("Unsupported vectorization method")
    return vectorizer.fit_transform(corpus).toarray(), vectorizer

# Train and Evaluate Models
def evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(),
        'KNN': KNeighborsClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'SVM': SVC(probability=True),
        'XGBoost': XGBClassifier()
    }

    results = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

        results[name] = {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'auc': auc
        }

        print(f"{name} - Accuracy: {accuracy}")
        if auc:
            print(f"{name} - AUC: {auc}")

    return results

# Main Function
def main():
    filepath = input("Enter the dataset filepath: ")
    data = load_dataset(filepath)

    print("\nCleaning text data...")
    corpus = clean_texts(data, column='Review', remove_stopwords=True)

    print("\nVectorizing text data...")
    method = input("Choose vectorization method (bow/tfidf): ")
    X, vectorizer = vectorize_texts(corpus, method)
    y = data.iloc[:, 1].values

    test_size = float(input("Enter test size (e.g., 0.2 for 20%): "))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    print("\nEvaluating models...")
    results = evaluate_models(X_train, X_test, y_train, y_test)

    print("\nResults Summary:")
    for model, metrics in results.items():
        print(f"\nModel: {model}")
        print(f"Accuracy: {metrics['accuracy']}")
        print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")
        if metrics['auc'] is not None:
            print(f"AUC: {metrics['auc']}")

if __name__ == "__main__":
    main()
