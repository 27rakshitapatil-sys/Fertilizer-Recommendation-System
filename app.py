from flask import Flask, render_template, request
import numpy as np
import pickle
import gzip

app = Flask(__name__)

# Load trained model and encoders
with gzip.open("model.pkl.gz", "rb") as file:
    model = pickle.load(file)
soil_encoder = pickle.load(open("soil_encoder.pkl", "rb"))
crop_encoder = pickle.load(open("crop_encoder.pkl", "rb"))
fert_encoder = pickle.load(open("fert_encoder.pkl", "rb"))


# Feature names used by the model
feature_names = [
    "Temperature",
    "Humidity",
    "Moisture",
    "Soil Type",
    "Crop Type",
    "Nitrogen",
    "Potassium",
    "Phosphorous"
]


@app.route("/")
def home():

    # Get actual values stored inside the encoders
    soil_types = soil_encoder.classes_.tolist()
    crop_types = crop_encoder.classes_.tolist()

    # Get actual Random Forest feature importance
    feature_importance = model.feature_importances_.tolist()

    return render_template(
        "index.html",
        soil_types=soil_types,
        crop_types=crop_types,
        feature_names=feature_names,
        feature_importance=feature_importance
    )


@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get form values
        temp = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        moisture = float(request.form["moisture"])

        soil = request.form["soil"]
        crop = request.form["crop"]

        nitrogen = float(request.form["nitrogen"])
        potassium = float(request.form["potassium"])
        phosphorous = float(request.form["phosphorous"])

        # Encode categorical values
        soil_encoded = soil_encoder.transform([soil])[0]
        crop_encoded = crop_encoder.transform([crop])[0]

        # Create feature array
        features = np.array([
            [
                temp,
                humidity,
                moisture,
                soil_encoded,
                crop_encoded,
                nitrogen,
                potassium,
                phosphorous
            ]
        ])

        # Make prediction
        prediction = model.predict(features)

        # Convert prediction back to fertilizer name
        fertilizer = fert_encoder.inverse_transform(prediction)[0]

        # Get prediction confidence
        probabilities = model.predict_proba(features)[0]
        confidence = float(np.max(probabilities) * 100)

        # Get all fertilizer names and their probabilities
        fertilizer_names = fert_encoder.inverse_transform(
            np.arange(len(probabilities))
        )

        probability_data = [
            {
                "name": str(name),
                "probability": round(float(probability) * 100, 2)
            }
            for name, probability in zip(fertilizer_names, probabilities)
            if probability > 0
        ]

        # Sort by probability
        probability_data.sort(
            key=lambda x: x["probability"],
            reverse=True
        )

        soil_types = soil_encoder.classes_.tolist()
        crop_types = crop_encoder.classes_.tolist()

        feature_importance = model.feature_importances_.tolist()

        return render_template(
            "index.html",
            prediction_text=fertilizer,
            confidence=round(confidence, 2),
            probability_data=probability_data,
            soil_types=soil_types,
            crop_types=crop_types,
            feature_names=feature_names,
            feature_importance=feature_importance,
            selected_soil=soil,
            selected_crop=crop
        )

    except Exception as e:

        soil_types = soil_encoder.classes_.tolist()
        crop_types = crop_encoder.classes_.tolist()
        feature_importance = model.feature_importances_.tolist()

        return render_template(
            "index.html",
            error_message="Please enter valid values and try again.",
            soil_types=soil_types,
            crop_types=crop_types,
            feature_names=feature_names,
            feature_importance=feature_importance
        )


if __name__ == "__main__":
    app.run(debug=False)