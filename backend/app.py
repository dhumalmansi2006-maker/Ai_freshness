from flask import Flask, request, jsonify
from flask_cors import CORS
import os

from predict_image import predict_image
from smell_analysis import analyze_smell
from final_prediction import get_final_prediction
from database import get_db_connection


app = Flask(__name__)
CORS(app)

# Folder for uploaded images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/")
def home():
    return "Freshness Detection API is Running!"


@app.route("/predict", methods=["POST"])
def predict():

    try:

        # -----------------------------
        # GET IMAGE
        # -----------------------------

        if "image" not in request.files:
            return jsonify({
                "error": "No image uploaded"
            }), 400

        image = request.files["image"]

        if image.filename == "":
            return jsonify({
                "error": "No image selected"
            }), 400

        # -----------------------------
        # GET SMELL DESCRIPTION
        # -----------------------------

        smell_description = request.form.get(
            "smell_description",
            ""
        )

        # Optional food name
        food_name = request.form.get(
            "food_name",
            "Unknown"
        )

        # -----------------------------
        # SAVE IMAGE
        # -----------------------------

        image_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            image.filename
        )

        image.save(image_path)

        # -----------------------------
        # IMAGE PREDICTION
        # -----------------------------

        image_result, image_confidence = predict_image(
            image_path
        )

        # -----------------------------
        # SMELL ANALYSIS
        # -----------------------------

        smell_result = analyze_smell(
            smell_description
        )

        # -----------------------------
        # FINAL PREDICTION
        # -----------------------------

        final_result = get_final_prediction(
            image_result,
            smell_result
        )

        # -----------------------------
        # SAVE TO MYSQL
        # -----------------------------

        connection = get_db_connection()
        cursor = connection.cursor()

        query = """
        INSERT INTO predictions
        (
            food_name,
            smell_description,
            image_result,
            final_result,
            confidence
        )
        VALUES (%s, %s, %s, %s, %s)
        """

        values = (
            food_name,
            smell_description,
            image_result,
            final_result,
            float(image_confidence)
        )

        cursor.execute(query, values)

        connection.commit()

        cursor.close()
        connection.close()

        # -----------------------------
        # RETURN RESULT
        # -----------------------------

        return jsonify({
            "food_name": food_name,
            "image_result": image_result,
            "image_confidence": round(
                float(image_confidence), 2
            ),
            "smell_result": smell_result,
            "final_result": final_result,
            "message": "Prediction saved successfully!"
        })

    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)