import os
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ==========================================
# SETTINGS
# ==========================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

TRAIN_DIR = "dataset/train"
VALIDATION_DIR = "dataset/validation"

MODEL_DIR = "model"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "freshness_model.keras"
)


# ==========================================
# CREATE MODEL DIRECTORY
# ==========================================

os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================
# LOAD TRAINING DATA
# ==========================================

print("\nLoading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
    seed=42
)


# ==========================================
# LOAD VALIDATION DATA
# ==========================================

print("\nLoading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False
)


# ==========================================
# DISPLAY CLASS NAMES
# ==========================================

class_names = train_dataset.class_names

print("\nClasses found:")
print(class_names)


# ==========================================
# PERFORMANCE OPTIMIZATION
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ==========================================
# DATA AUGMENTATION
# ==========================================

data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="data_augmentation")


# ==========================================
# LOAD MOBILE NET V2
# ==========================================

print("\nLoading MobileNetV2...")

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)


# ==========================================
# FREEZE BASE MODEL
# ==========================================

base_model.trainable = False


# ==========================================
# BUILD OUR MODEL
# ==========================================

inputs = layers.Input(
    shape=(224, 224, 3)
)

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(
    x,
    training=False
)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    3,
    activation="softmax"
)(x)


model = models.Model(
    inputs,
    outputs
)


# ==========================================
# COMPILE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# DISPLAY MODEL
# ==========================================

print("\nModel summary:\n")

model.summary()


# ==========================================
# CALLBACKS
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)


model_checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)


# ==========================================
# TRAIN MODEL
# ==========================================

print("\nStarting training...\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        model_checkpoint
    ]
)


# ==========================================
# FINAL RESULTS
# ==========================================

print("\nTraining completed!")

print(
    f"\nBest model saved at:"
    f"\n{MODEL_PATH}"
)


# ==========================================
# SHOW FINAL ACCURACY
# ==========================================

final_train_accuracy = history.history[
    "accuracy"
][-1]

final_validation_accuracy = history.history[
    "val_accuracy"
][-1]


print(
    f"\nFinal Training Accuracy:"
    f" {final_train_accuracy * 100:.2f}%"
)

print(
    f"Final Validation Accuracy:"
    f" {final_validation_accuracy * 100:.2f}%"
)