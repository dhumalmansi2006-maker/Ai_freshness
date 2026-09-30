import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


# ==========================================
# SETTINGS
# ==========================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

TEST_DIR = "dataset/test"
MODEL_PATH = "model/freshness_model.keras"


# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==========================================
# LOAD TEST DATASET
# ==========================================

print("\nLoading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)


# ==========================================
# CLASS NAMES
# ==========================================

class_names = test_dataset.class_names

print("\nClasses:")
print(class_names)


# ==========================================
# GET TRUE LABELS
# ==========================================

true_labels = []

for images, labels in test_dataset:

    labels = labels.numpy()

    true_labels.extend(
        np.argmax(labels, axis=1)
    )


true_labels = np.array(true_labels)


# ==========================================
# MAKE PREDICTIONS
# ==========================================

print("\nMaking predictions...")

predictions = model.predict(test_dataset)

predicted_labels = np.argmax(
    predictions,
    axis=1
)


# ==========================================
# ACCURACY
# ==========================================

accuracy = np.mean(
    predicted_labels == true_labels
)

print("\n" + "=" * 50)

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)

print("=" * 50)


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:\n")

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

print("\nConfusion Matrix:\n")

matrix = confusion_matrix(
    true_labels,
    predicted_labels
)

print(matrix)