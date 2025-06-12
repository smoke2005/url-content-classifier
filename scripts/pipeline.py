# app.py (Flask Backend)
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import re
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

# === Constants ===
IMG_SIZE = (224, 224)
CLASS_NAMES = ["gambling", "non_gambling"]

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains on all routes

# === Utility: Clean Text ===
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# === Routes ===

@app.route("/predict/hate", methods=["POST"])
def predict_hatespeech():
    text = request.json.get("text", "")
    model = joblib.load('./models/hatespeech_classifier_xgb.pkl')
    cleaned = clean_text(text)
    prediction = model.predict([cleaned])[0]
    return jsonify({"label": "hate" if prediction == 1 else "non-hate"})

@app.route("/predict/phishing", methods=["POST"])
def predict_url_phishing():
    url = request.json.get("url", "")
    model = joblib.load('./models/url_phishing_model.pkl')
    pred = model.predict([url])[0]
    prob = model.predict_proba([url])[0]
    label = "phishing" if pred == 1 else "legitimate"
    confidence = float(max(prob))
    return jsonify({"label": label, "confidence": confidence})

@app.route("/predict/adult", methods=["POST"])
def predict_explicit_content():
    text = request.json.get("text", "")
    model = joblib.load('./models/adult_classifier.pkl')
    vectorizer = joblib.load('./models/tfidf_vectorizer.pkl')
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return jsonify({"label": "adult" if pred == 1 else "non-adult"})

@app.route("/predict/gambling", methods=["POST"])
def predict_image_gambling():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded."}), 400
    
    file = request.files['image']
    img_path = os.path.join("uploads", file.filename)
    file.save(img_path)

    model = load_model('./models/densenet_gambling_classifier_augmented.h5')
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = CLASS_NAMES[int(prediction > 0.5)]
    confidence = float(prediction if prediction > 0.5 else 1 - prediction)

    return jsonify({"label": label, "confidence": confidence})

if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.mkdir("uploads")
    app.run(debug=True)
