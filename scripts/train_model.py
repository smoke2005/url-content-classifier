import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

def train_and_save_model(dataset_path):
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=['Category', 'Description'])
    df['Category'] = df['Category'].str.strip().str.lower()
    df['Category'] = df['Category'].replace({
        'non_adult': 'non-adult',
        'nonadult': 'non-adult',
        'non adult': 'non-adult',
        'adult': 'adult'
    })
    df['label'] = df['Category'].map({'adult': 1, 'non-adult': 0})
    df = df.dropna(subset=['label'])

    print("\nClass distribution:")
    print(df['label'].value_counts())

    if df['label'].nunique() < 2:
        raise ValueError("Dataset must contain both 'adult' and 'non-adult' classes.")

    X_train, X_test, y_train, y_test = train_test_split(df['Description'], df['label'], test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    print("\nEvaluation Metrics")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, '../models/adult_classifier.pkl')
    joblib.dump(vectorizer, '../models/tfidf_vectorizer.pkl')


if __name__ == "__main__":
    dataset_path = "../data/Copy of final_adult_content(1).csv"
    train_and_save_model(dataset_path)
