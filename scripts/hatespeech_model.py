import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.utils import resample
import joblib

# === Load and Clean Dataset ===
df = pd.read_csv("C:\\Users\\mokit\\Downloads\\harmful_content_collection_scripts\\data\\HateSpeechDataset.csv")

# Drop non-numeric labels
df = df[df['Label'].apply(lambda x: str(x).isdigit())]
df['Label'] = df['Label'].astype(int)

# === Balance Dataset ===
df_majority = df[df.Label == 0]
df_minority = df[df.Label == 1]

df_majority_downsampled = resample(df_majority, replace=False,
                                   n_samples=len(df_minority),
                                   random_state=42)
df_balanced = pd.concat([df_majority_downsampled, df_minority])
df_balanced = df_balanced.sample(frac=1, random_state=42)  # Shuffle

# === Text Preprocessing ===
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

df_balanced["Content"] = df_balanced["Content"].apply(clean_text)

# === Split Dataset ===
X_train, X_test, y_train, y_test = train_test_split(
    df_balanced["Content"], df_balanced["Label"],
    test_size=0.2, random_state=42, stratify=df_balanced["Label"]
)

# === Vectorizers ===
word_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=10000)
char_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=5000)

combined_vectorizer = FeatureUnion([
    ('word', word_vectorizer),
    ('char', char_vectorizer)
])

# === Model Pipeline ===
pipeline = Pipeline([
    ('vectorizer', combined_vectorizer),
    ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

# === Train ===
pipeline.fit(X_train, y_train)

# === Evaluate ===
y_pred = pipeline.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))

# === Save Model ===
joblib.dump(pipeline, 'hatespeech_classifier_xgb.pkl')  