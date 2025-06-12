import joblib
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

IMG_SIZE = (224, 224)
class_names = ["gambling", "non_gambling"]

def predict_image(model, img_path, class_names=["gambling", "non_gambling"]):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    label = class_names[int(prediction > 0.6)]
    confidence = prediction if prediction > 0.6 else 1 - prediction

    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Prediction: {label} ({confidence:.2%} confidence)")
    plt.show()

    return label, confidence

# Test prediction
model = load_model('densenet_gambling_classifier_augmented.h5')
img_path = "C:\\Users\\mokit\\Downloads\\harmful_content_collection_scripts\\scripts\\screenshots\\page_ss.png"
label, confidence = predict_image(model, img_path, class_names)
print("Predicted Label:", label)
print("Confidence:", confidence)