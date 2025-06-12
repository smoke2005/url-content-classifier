from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import joblib

# Load data
df = pd.read_csv("C:\\Users\\mokit\\Downloads\\harmful_content_collection_scripts\\data\\dataset_phishing.csv")
df['status'] = df['status'].map({'phishing': 1, 'legitimate': 0})

# Split
X_train, X_test, y_train, y_test = train_test_split(df['url'], df['status'], test_size=0.25, random_state=42)

# Build pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5))),  # character-level n-grams
    ('clf', LogisticRegression())
])

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(model, '../model/url_phishing_model.pkl')
