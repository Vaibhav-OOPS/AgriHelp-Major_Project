# -------------------------------
# Imports
# -------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# -------------------------------
# Load dataset
# -------------------------------
data = pd.read_csv('Fertilizer_Prediction.csv')  # CSV with features + Fertilizer

# -------------------------------
# Encode categorical features manually
# -------------------------------
# Soil mapping
soil_classes = data['Soil'].unique().tolist()
soil_mapping = {cls: idx for idx, cls in enumerate(soil_classes)}

# Crop mapping
crop_classes = data['Crop'].unique().tolist()
crop_mapping = {cls: idx for idx, cls in enumerate(crop_classes)}

# Fertilizer mapping (target)
fertilizer_classes = data['Fertilizer'].unique().tolist()
fertilizer_mapping = {cls: idx for idx, cls in enumerate(fertilizer_classes)}
fertilizer_reverse_mapping = {idx: cls for cls, idx in fertilizer_mapping.items()}

# Encode dataset
data['Soil'] = data['Soil'].map(soil_mapping)
data['Crop'] = data['Crop'].map(crop_mapping)
data['Fertilizer'] = data['Fertilizer'].map(fertilizer_mapping)

# -------------------------------
# Split features and target
# -------------------------------
X = data[['Temperature','Humidity','Rainfall','pH','N','P','K','Soil','Crop']]
y = data['Fertilizer']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Random Forest
# -------------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# -------------------------------
# Evaluate model
# -------------------------------
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# -------------------------------
# Save model and mappings
# -------------------------------
with open('rf_fertilizer_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

with open('soil_mapping.pkl', 'wb') as f:
    pickle.dump(soil_mapping, f)

with open('crop_mapping.pkl', 'wb') as f:
    pickle.dump(crop_mapping, f)

with open('fertilizer_mapping.pkl', 'wb') as f:
    pickle.dump(fertilizer_reverse_mapping, f)

print("Model and mappings saved successfully!")
