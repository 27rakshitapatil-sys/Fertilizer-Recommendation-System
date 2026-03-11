from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
soil_encoder = pickle.load(open("soil_encoder.pkl","rb"))
crop_encoder = pickle.load(open("crop_encoder.pkl","rb"))
fert_encoder = pickle.load(open("fert_encoder.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    temp = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    moisture = float(request.form["moisture"])
    soil = request.form["soil"]
    crop = request.form["crop"]
    nitrogen = float(request.form["nitrogen"])
    potassium = float(request.form["potassium"])
    phosphorous = float(request.form["phosphorous"])

    soil_encoded = soil_encoder.transform([soil])[0]
    crop_encoded = crop_encoder.transform([crop])[0]

    features = np.array([[temp,humidity,moisture,
                          soil_encoded,crop_encoded,
                          nitrogen,potassium,phosphorous]])

    prediction = model.predict(features)
    fertilizer = fert_encoder.inverse_transform(prediction)

    return render_template("index.html",
                           prediction_text="Recommended Fertilizer: "+fertilizer[0])

if __name__ == "__main__":
    app.run(debug=True)