import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv("crops_quality_and_farmers_friend.csv")

print("Dataset Loaded Successfully")

print("Dataset Shape:", data.shape)

data.head()
print(data.columns)
data.info()
data.describe()
plt.figure(figsize=(10,6))

sns.countplot(x="Fertilizer Name", data=data)

plt.title("Fertilizer Distribution")

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(8,5))

sns.countplot(x="Soil Type", data=data)

plt.title("Soil Type Distribution")

plt.show()
plt.figure(figsize=(8,5))

sns.countplot(x="Crop Type", data=data)

plt.title("Crop Type Distribution")

plt.show()
plt.figure(figsize=(8,5))

sns.scatterplot(
    x="Temparature",
    y="Humidity",
    hue="Fertilizer Name",
    data=data
)

plt.title("Temperature vs Humidity")

plt.show()
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
sns.histplot(data["Nitrogen"], kde=True)
plt.title("Nitrogen Distribution")

plt.subplot(1,3,2)
sns.histplot(data["Potassium"], kde=True)
plt.title("Potassium Distribution")
numeric_data = data.select_dtypes(include=np.number)

plt.figure(figsize=(10,6))

sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")

plt.title("Feature Correlation Heatmap")

plt.show()
soil_encoder = LabelEncoder()
crop_encoder = LabelEncoder()
fert_encoder = LabelEncoder()

data["Soil Type"] = soil_encoder.fit_transform(data["Soil Type"])
data["Crop Type"] = crop_encoder.fit_transform(data["Crop Type"])
data["Fertilizer Name"] = fert_encoder.fit_transform(data["Fertilizer Name"])

data.head()
X = data.drop("Fertilizer Name", axis=1)

y = data["Fertilizer Name"]

print("Feature Shape:", X.shape)
print("Target Shape:", y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)
models = {

    "Decision Tree": DecisionTreeClassifier(),

    "Random Forest": RandomForestClassifier(n_estimators=200),

    "SVM": SVC(),

    "KNN": KNeighborsClassifier()

}

results = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    acc = accuracy_score(y_test, pred)

    results[name] = acc

    print("Model:", name)
    print("Accuracy:", acc)
    print("--------------------")
    plt.figure(figsize=(8,5))

sns.barplot(
    x=list(results.keys()),
    y=list(results.values())
)

plt.title("Model Accuracy Comparison")

plt.ylabel("Accuracy")

plt.show()
final_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

final_model.fit(X_train, y_train)

predictions = final_model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Final Model Accuracy:", accuracy)
cm = confusion_matrix(y_test, predictions)

plt.figure(figsize=(8,6))

sns.heatmap(cm, annot=True, cmap="Blues")

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()
importance = final_model.feature_importances_

features = X.columns

plt.figure(figsize=(10,6))

sns.barplot(x=importance, y=features)

plt.title("Feature Importance")

plt.show()
def predict_fertilizer():

    temp = float(input("Enter Temperature: "))
    humidity = float(input("Enter Humidity: "))
    moisture = float(input("Enter Moisture: "))

    soil = input("Enter Soil Type: ")
    crop = input("Enter Crop Type: ")

    nitrogen = float(input("Enter Nitrogen: "))
    potassium = float(input("Enter Potassium: "))
    phosphorous = float(input("Enter Phosphorous: "))

    soil_encoded = soil_encoder.transform([soil])[0]
    crop_encoded = crop_encoder.transform([crop])[0]

    features = np.array([[
        temp,
        humidity,
        moisture,
        soil_encoded,
        crop_encoded,
        nitrogen,
        potassium,
        phosphorous
    ]])

    prediction = final_model.predict(features)

    fertilizer = fert_encoder.inverse_transform(prediction)


    print("Recommended Fertilizer:", fertilizer[0])
    
# predict_fertilizer()   # disable terminal prediction

import pickle

pickle.dump(final_model, open("model.pkl", "wb"))
pickle.dump(soil_encoder, open("soil_encoder.pkl", "wb"))
pickle.dump(crop_encoder, open("crop_encoder.pkl", "wb"))
pickle.dump(fert_encoder, open("fert_encoder.pkl", "wb"))

print("Model Saved Successfully")