import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os

# === Config ===
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
train_path = "../data/train"
val_path = "../data/val"
test_path = "./data/test"

# === Data Augmentation ===
data_augmentation = tf.keras.Sequential([
    layers.Rescaling(1./255),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.2),
    layers.RandomContrast(0.2),
])

# === Load and Balance Dataset ===
def make_balanced_augmented_dataset(train_path, img_size, batch_size):
    full_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        labels='inferred',
        label_mode='int',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=42
    ).unbatch()

    # Separate by class
    gambling_ds = full_ds.filter(lambda x, y: tf.equal(y, 1))
    non_gambling_ds = full_ds.filter(lambda x, y: tf.equal(y, 0))

    # Get number of minority samples
    gambling_count = gambling_ds.reduce(0, lambda x, _: x + 1).numpy()
    non_gambling_count = non_gambling_ds.reduce(0, lambda x, _: x + 1).numpy()
    min_count = min(gambling_count, non_gambling_count)

    # Balance the classes
    gambling_ds = gambling_ds.take(min_count)
    non_gambling_ds = non_gambling_ds.take(min_count)

    # Apply augmentation and rescaling
    gambling_ds = gambling_ds.map(lambda x, y: (data_augmentation(x), y))
    non_gambling_ds = non_gambling_ds.map(lambda x, y: (data_augmentation(x), y))

    combined_ds = (gambling_ds.concatenate(non_gambling_ds)
                   .shuffle(1000)
                   .batch(batch_size)
                   .prefetch(tf.data.AUTOTUNE))

    return combined_ds

# === Compute Class Weights (Optional) ===
def compute_class_weights(train_path):
    raw_ds = tf.keras.utils.image_dataset_from_directory(
        train_path,
        label_mode='int',
        image_size=IMG_SIZE,
        batch_size=None
    )

    labels = []
    for _, y in raw_ds:
        labels.append(y.numpy())

    labels = np.array(labels).flatten()
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    return {i: w for i, w in enumerate(weights)}

# === Create datasets ===
train_ds = make_balanced_augmented_dataset(train_path, IMG_SIZE, BATCH_SIZE)
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
).map(lambda x, y: (x / 255.0, y)).prefetch(tf.data.AUTOTUNE)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
).map(lambda x, y: (x / 255.0, y)).prefetch(tf.data.AUTOTUNE)

class_weights = compute_class_weights(train_path)

# === Build Model ===
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# === Train ===
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stop],
    class_weight=class_weights
)

# === Evaluate ===
loss, acc, auc = model.evaluate(test_ds)
print(f"\nTest Accuracy: {acc:.2%}\n")

# === Confusion Matrix and Classification Report ===
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend((preds > 0.5).astype(int).flatten())

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Non-Gambling', 'Gambling']))

# === Save Model ===
model.save('../models/densenet_gambling_classifier_augmented.h5')
