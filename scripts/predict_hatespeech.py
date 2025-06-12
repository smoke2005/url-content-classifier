import joblib
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def predict_hatespeech(text):
    # Load pipeline (includes vectorizer + XGBoost model)
    model = joblib.load('../models/hatespeech_classifier_xgb.pkl')
    
    # Clean and predict
    cleaned = clean_text(text)
    prediction = model.predict([cleaned])[0]
    return "hate" if prediction == 1 else "non-hate"

if __name__ == "__main__":
    with open('page_text.txt', 'r', encoding='utf-8') as f:
        scraped_text = f.read()

    result = predict_hatespeech(scraped_text)
    print(scraped_text)
    print("\nPrediction for scraped page:", result)
