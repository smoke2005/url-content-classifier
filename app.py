from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64
import os
import joblib
import re
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from scrape_website import scrape_and_save

IMG_SIZE = (224, 224)
CLASS_NAMES = ["gambling", "non_gambling"]
TEXT_PATH = "page_text.txt"
IMAGE_PATH = "page_ss.png"

app = Flask(__name__)
CORS(app)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def predict_hatespeech(text):
    model = joblib.load('models/hatespeech_classifier_xgb.pkl')
    cleaned = clean_text(text)
    proba = model.predict_proba([cleaned])[0]
    pred = model.predict([cleaned])[0]
    label = "Hate" if pred == 1 else "Non-hate"
    confidence = float(proba[1]) if pred == 1 else float(proba[0])
    return label, confidence

def predict_url_phishing(url):
    model = joblib.load('models/url_phishing_model.pkl')
    pred = model.predict([url])[0]
    prob = model.predict_proba([url])[0]
    label = "Phishing" if pred == 1 else "Legitimate"
    confidence = float(max(prob))
    return label, confidence

def predict_explicit_content(text):
    model = joblib.load('models/adult_classifier.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    proba = model.predict_proba(vec)[0]
    pred = model.predict(vec)[0]
    label = "Adult" if pred == 1 else "Non-adult"
    confidence = float(proba[1]) if pred == 1 else float(proba[0])
    return label, confidence

def predict_image_gambling(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at: {img_path}")
    model = load_model('models/densenet_gambling_classifier_augmented.h5')
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    label = CLASS_NAMES[int(prediction > 0.5)]
    confidence = float(prediction if prediction > 0.5 else 1 - prediction)
    return label, confidence

def get_final_safety_verdict(hate, explicit, phishing, gambling):
    print(f"[DEBUG] Hate: {hate}, Explicit: {explicit}, Phishing: {phishing}, Gambling: {gambling}")
    unsafe_triggers = ["hate", "adult", "phishing", "gambling"]
    if hate == "Hate" or explicit == "Adult" or phishing == "Phishing" or gambling == "gambling":
        return "Unsafe"
    return "Safe"


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        url = data.get("url")
        if not url:
            return jsonify({"error": "URL is required"}), 400

        scrape_and_save(url, output_text=TEXT_PATH, output_image=IMAGE_PATH)

        with open(TEXT_PATH, "r", encoding="utf-8") as f:
            scraped_text = f.read()

        hate_result, hate_conf = predict_hatespeech(scraped_text)
        explicit_result, explicit_conf = predict_explicit_content(scraped_text)
        phishing_result, phishing_conf = predict_url_phishing(url)
        gambling_result, gambling_conf = predict_image_gambling(IMAGE_PATH)

        with open("page_ss.png", "rb") as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        safety = get_final_safety_verdict(hate_result, explicit_result, phishing_result, gambling_result)

        return jsonify({
            "hate_speech": hate_result,
            "hate_conf": hate_conf,
            "explicit": explicit_result,
            "explicit_conf": explicit_conf,
            "phishing": phishing_result,
            "phishing_conf": phishing_conf,
            "gambling": gambling_result,
            "gambling_conf": gambling_conf,
            "safety_verdict": safety,
            "image_base64": image_base64
        })

    except Exception as e:
        print("[ERROR]", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
