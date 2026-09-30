import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

MODEL_PATH = "model/freshness_model.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Fresh", "Semi-Fresh", "Spoiled"]

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")


def predict_image(image_path):

    image = load_img(
        image_path,
        target_size=IMG_SIZE
    )

    image_array = img_to_array(image)

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    predictions = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = np.argmax(predictions[0])

    confidence = predictions[0][predicted_index] * 100

    predicted_class = CLASS_NAMES[predicted_index]

    return predicted_class, confidence


# Run interactive testing ONLY when this file
# is executed directly.
# Flask will NOT execute this section when importing it.
if __name__ == "__main__":

    image_path = input(
        "\nEnter the path of the image: "
    )

    try:

        result, confidence = predict_image(
            image_path
        )

        print("\n" + "=" * 40)
        print(f"Freshness: {result}")
        print(f"Confidence: {confidence:.2f}%")
        print("=" * 40)

    except Exception as error:

        print("\nError while predicting:")
        print(error)