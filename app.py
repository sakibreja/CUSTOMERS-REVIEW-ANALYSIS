from flask import Flask, request, render_template, jsonify
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Fetch the uploaded file and user inputs
    file = request.files['dataset']
    vectorizer_type = request.form['vectorizer']
    test_size = float(request.form['test_size'])

    # Read and preprocess the dataset
    dataset = pd.read_csv(file, delimiter='\t', quoting=3)
    corpus = []
    ps = PorterStemmer()
    nltk.download('stopwords')
    for review in dataset['Review']:
        review = re.sub('[^a-zA-Z]', ' ', review)
        review = review.lower().split()
        review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
        corpus.append(' '.join(review))

    # Vectorization
    if vectorizer_type == 'bow':
        vectorizer = CountVectorizer()
    else:
        vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus).toarray()
    y = dataset.iloc[:, 1].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    # Train and evaluate models
    classifiers = {
        'Logistic Regression': LogisticRegression(),
        'KNN': KNeighborsClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Decision Tree': DecisionTreeClassifier(),
        'SVM': SVC(probability=True),
        'XGBoost': XGBClassifier(),
    }
    results = {}
    for name, model in classifiers.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred).tolist()
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model, "predict_proba") else None
        results[name] = {
            "accuracy": round(acc, 3),
            "auc": round(auc, 3) if auc is not None else None,
            #"confusion_matrix": cm
        }

    # Return the results in the desired format
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
