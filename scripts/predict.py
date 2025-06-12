import joblib

def predict_adult(text):
    vectorizer = joblib.load('../models/tfidf_vectorizer.pkl')
    model = joblib.load('../models/adult_classifier.pkl')
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return "adult" if pred == 1 else "non-adult"

if __name__ == "__main__":
    with open('page_text.txt', 'r', encoding='utf-8') as f:
        scraped_text = f.read()

    result = predict_adult(scraped_text)
    print("\nPrediction for scraped page:", result)
