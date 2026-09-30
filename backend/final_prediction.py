from predict_image import predict_image
from smell_analysis import analyze_smell


def get_final_prediction(image_result, smell_result):
    """
    Combine image prediction and smell prediction.
    """

    # If either the image or smell indicates spoiled food,
    # classify the food as spoiled.
    if image_result == "Spoiled" or smell_result == "Spoiled":
        return "Spoiled"

    # If both indicate fresh, classify as fresh.
    if image_result == "Fresh" and smell_result == "Fresh":
        return "Fresh"

    # All other combinations are considered semi-fresh.
    return "Semi-Fresh"


if __name__ == "__main__":

    print("=" * 50)
    print("     AI FRESHNESS DETECTION")
    print("=" * 50)

    # --------------------------------
    # IMAGE INPUT
    # --------------------------------

    image_path = input("\nEnter the path of the image: ")

    try:
        image_result, image_confidence = predict_image(image_path)

        print("\nImage Prediction:", image_result)
        print(f"Image Confidence: {image_confidence:.2f}%")

    except Exception as error:
        print("\nError while predicting image:")
        print(error)
        exit()

    # --------------------------------
    # SMELL INPUT
    # --------------------------------

    smell_description = input("\nEnter smell description: ")

    smell_result = analyze_smell(smell_description)

    print("Smell Prediction:", smell_result)

    # --------------------------------
    # FINAL PREDICTION
    # --------------------------------

    final_result = get_final_prediction(
        image_result,
        smell_result
    )

    print("\n" + "=" * 50)
    print("FINAL RESULT")
    print("=" * 50)

    print("Image Result      :", image_result)
    print("Image Confidence  :", f"{image_confidence:.2f}%")
    print("Smell Result      :", smell_result)
    print("Final Freshness   :", final_result)

    print("=" * 50)