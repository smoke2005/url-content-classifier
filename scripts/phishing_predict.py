import joblib

# Load saved model
model = joblib.load('../models/url_phishing_model.pkl')

def predict_url_phishing(url):
    pred = model.predict([url])[0]
    prob = model.predict_proba([url])[0]
    print(f"Prediction: {'Phishing' if pred == 1 else 'Legitimate'}")
    print(f"Confidence: Legitimate={prob[0]:.4f}, Phishing={prob[1]:.4f}")
    return pred

# Example usage
pred=predict_url_phishing("https://stripchat.global/")
print(pred)