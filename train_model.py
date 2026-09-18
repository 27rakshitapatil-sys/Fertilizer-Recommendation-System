import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# Load dataset
data = pd.read_csv("crops_quality_and_farmers_friend.csv")

print("Dataset Loaded Successfully")
print("Dataset Shape:", data.shape)


# Create encoders
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
fert_encoder = LabelEncoder()


# Encode categorical columns
data["Soil Type"] = soil_encoder.fit_transform(data["Soil Type"])
data["Crop Type"] = crop_encoder.fit_transform(data["Crop Type"])
data["Fertilizer Name"] = fert_encoder.fit_transform(data["Fertilizer Name"])


# Features and target
X = data.drop("Fertilizer Name", axis=1)
y = data["Fertilizer Name"]

print("Feature Shape:", X.shape)
print("Target Shape:", y.shape)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)


# Train Random Forest model
# 100 trees reduces memory usage for deployment
final_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print("Training Random Forest model...")


final_model.fit(X_train, y_train)


# Evaluate model
predictions = final_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Final Model Accuracy:", accuracy)


# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(final_model, file)


# Save encoders
with open("soil_encoder.pkl", "wb") as file:
    pickle.dump(soil_encoder, file)

with open("crop_encoder.pkl", "wb") as file:
    pickle.dump(crop_encoder, file)

with open("fert_encoder.pkl", "wb") as file:
    pickle.dump(fert_encoder, file)


print("Model Saved Successfully")
print("Created: model.pkl")
print("Created: soil_encoder.pkl")
print("Created: crop_encoder.pkl")
print("Created: fert_encoder.pkl")