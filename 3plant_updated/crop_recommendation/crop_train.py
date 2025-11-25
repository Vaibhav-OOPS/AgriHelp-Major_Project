    # ============================================================
# CROP RECOMMENDATION + MARKET PRICE PREDICTION (FINAL)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# ------------------------------------------------------------
# 1️⃣ LOAD DATA
# ------------------------------------------------------------
crop = pd.read_csv("Crop_recommendation.csv")
print("Dataset Loaded Successfully ✅")

print("\nFirst 5 Rows:")
print(crop.head())

print("\nMissing Values:")
print(crop.isnull().sum())

# ------------------------------------------------------------
# 2️⃣ CLEANING MISSING VALUES
# ------------------------------------------------------------
# Fill or drop missing marketing price values
if 'marketing price' in crop.columns:
    crop['marketing price'].fillna(crop['marketing price'].mean(), inplace=True)

# ------------------------------------------------------------
# 3️⃣ ENCODE LABELS AND ADD MARKET PRICE COLUMN
# ------------------------------------------------------------
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5,
    'papaya': 6, 'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10,
    'grapes': 11, 'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15,
    'blackgram': 16, 'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19,
    'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
}

crop['label'] = crop['label'].map(crop_dict)

# Market price mapping (in ₹/kg)
price_dict = {
    1: 34, 2: 25, 3: 82.75, 4: 68.5, 5: 30, 6: 52.5, 7: 55, 8: 105, 9: 49.5, 10: 22,
    11: 100, 12: 95.14, 13: 34.61, 14: 120, 15: 67.81, 16: 63.05, 17: 97.24, 18: 65,
    19: 82.33, 20: 100, 21: 98, 22: 17.6
}

crop['market_price'] = crop['label'].map(price_dict)

# ------------------------------------------------------------
# 4️⃣ DEFINE FEATURES AND TARGET
# ------------------------------------------------------------
# Ensure correct feature selection
if 'marketing price' in crop.columns:
    X = crop.drop(['label', 'market_price', 'marketing price'], axis=1, errors='ignore')
else:
    X = crop.drop(['label', 'market_price'], axis=1, errors='ignore')

y = crop['label']

# ------------------------------------------------------------
# 5️⃣ TRAIN-TEST SPLIT AND SCALING
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mx = MinMaxScaler()
sc = StandardScaler()

X_train = mx.fit_transform(X_train)
X_test = mx.transform(X_test)

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ------------------------------------------------------------
# 6️⃣ TRAIN MODEL
# ------------------------------------------------------------
model = LogisticRegression(max_iter=300)
model.fit(X_train, y_train)

# ------------------------------------------------------------
# 7️⃣ EVALUATION
# ------------------------------------------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ------------------------------------------------------------
# 8️⃣ SAVE MODEL AND SCALERS
# ------------------------------------------------------------
pickle.dump(model, open('best_model.pkl', 'wb'))
pickle.dump(mx, open('minmaxscaler.pkl', 'wb'))
pickle.dump(sc, open('standscaler.pkl', 'wb'))

print("\nModel and Scalers Saved Successfully ✅")

# ------------------------------------------------------------
# 9️⃣ PREDICTION FUNCTION (Crop + Market Price)
# ------------------------------------------------------------
reverse_crop_dict = {v: k for k, v in crop_dict.items()}

def recommendation(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    mx_features = mx.transform(features)
    sc_features = sc.transform(mx_features)
    prediction = model.predict(sc_features)[0]
    crop_name = reverse_crop_dict[prediction]
    market_price = price_dict[prediction]
    return crop_name, market_price

# ------------------------------------------------------------
# 🔟 SAMPLE PREDICTION
# ------------------------------------------------------------
N, P, K = 8, 28, 38
temperature, humidity, ph, rainfall = 23.22, 94.43, 6.84, 105.69

pred_crop, pred_price = recommendation(N, P, K, temperature, humidity, ph, rainfall)
print(f"\nRecommended Crop: {pred_crop}")
print(f"Market Price (₹/kg): {pred_price}")
