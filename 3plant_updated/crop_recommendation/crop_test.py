import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import numpy as np
import pickle
import os

# ============================================================
# LOAD MODEL AND SCALERS
# ============================================================
model = pickle.load(open('best_model.pkl', 'rb'))
mx = pickle.load(open('minmaxscaler.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))

# ------------------------------------------------------------
# DICTIONARIES
# ------------------------------------------------------------
crop_dict = {
    1: 'rice', 2: 'maize', 3: 'jute', 4: 'cotton', 5: 'coconut',
    6: 'papaya', 7: 'orange', 8: 'apple', 9: 'muskmelon', 10: 'watermelon',
    11: 'grapes', 12: 'mango', 13: 'banana', 14: 'pomegranate', 15: 'lentil',
    16: 'blackgram', 17: 'mungbean', 18: 'mothbeans', 19: 'pigeonpeas',
    20: 'kidneybeans', 21: 'chickpea', 22: 'coffee'
}

price_dict = {
    1: 34, 2: 25, 3: 82.75, 4: 68.5, 5: 30, 6: 52.5, 7: 55, 8: 105, 9: 49.5, 10: 22,
    11: 100, 12: 95.14, 13: 34.61, 14: 120, 15: 67.81, 16: 63.05, 17: 97.24, 18: 65,
    19: 82.33, 20: 100, 21: 98, 22: 17.6
}

# ------------------------------------------------------------
# EXPLANATION MESSAGES (4 lines per crop)
# ------------------------------------------------------------
explanation_dict = {
    "rice": [
        "Rice thrives in high humidity and warm temperatures.",
        "Nitrogen and phosphorus levels support leaf growth and tillering.",
        "The soil pH and rainfall are ideal for paddy cultivation.",
        "Consistent water supply ensures high yield."
    ],
    "maize": [
        "Maize prefers moderate rainfall and warm climate.",
        "Balanced NPK levels enhance cob and kernel development.",
        "Good temperature and humidity support rapid growth.",
        "Soil nutrients match the maize crop’s nitrogen demand."
    ],
    "jute": [
        "Jute grows best in warm and humid conditions.",
        "High nitrogen promotes fiber quality and yield.",
        "Phosphorus and potassium improve stem strength.",
        "Adequate rainfall ensures proper retting and growth."
    ],
    "cotton": [
        "Cotton grows best in warm and dry climates.",
        "Slightly alkaline pH suits root development.",
        "Moderate rainfall prevents fungal issues.",
        "High potassium boosts fiber quality and yield."
    ],
    "coconut": [
        "Coconut palms prefer tropical climates with high humidity.",
        "Balanced NPK supports nut formation and leaf growth.",
        "Well-drained soil prevents root rot.",
        "Consistent rainfall helps steady fruiting."
    ],
    "papaya": [
        "Papaya thrives in warm, frost-free environments.",
        "High nitrogen encourages vegetative growth.",
        "Potassium and phosphorus improve fruit quality.",
        "Proper rainfall and humidity prevent diseases."
    ],
    "orange": [
        "Oranges require subtropical climate with moderate humidity.",
        "Nitrogen and potassium support fruit size and sweetness.",
        "Phosphorus aids in flowering and fruit set.",
        "Well-drained soil prevents root diseases."
    ],
    "apple": [
        "Apple grows best in temperate regions with cool winters.",
        "Adequate nitrogen and potassium enhance fruit quality.",
        "Balanced phosphorus supports flowering.",
        "Proper temperature and rainfall ensure fruit development."
    ],
    "muskmelon": [
        "Muskmelon prefers warm and sunny conditions.",
        "High potassium improves fruit sweetness and size.",
        "Nitrogen supports healthy vine growth.",
        "Moderate rainfall ensures optimal fruit formation."
    ],
    "watermelon": [
        "Watermelon thrives in warm, sunny climates.",
        "Balanced NPK supports vine and fruit growth.",
        "Soil with good drainage prevents waterlogging.",
        "Adequate rainfall ensures juicy and sweet fruits."
    ],
    "grapes": [
        "Grapes prefer warm, dry climate with good sunlight.",
        "Nitrogen promotes leaf growth, potassium improves fruit quality.",
        "Phosphorus supports flowering and fruit set.",
        "Moderate rainfall prevents fungal diseases."
    ],
    "mango": [
        "Mango trees love warm, sunny climates.",
        "Moderate nitrogen and phosphorus favor flowering.",
        "Good drainage and low humidity prevent fruit rot.",
        "Rainfall levels are ideal for fruit setting."
    ],
    "banana": [
        "Banana thrives in warm, humid environments.",
        "High nitrogen and potassium support leaf and fruit growth.",
        "Balanced pH ensures healthy root absorption.",
        "Consistent rainfall aids steady fruiting."
    ],
    "pomegranate": [
        "Pomegranate grows well in warm, dry climates.",
        "Moderate nitrogen encourages fruit set.",
        "Potassium improves fruit quality and sweetness.",
        "Low humidity reduces fungal disease risk."
    ],
    "lentil": [
        "Lentils prefer cool and moderately dry conditions.",
        "Nitrogen fixation by roots reduces fertilizer need.",
        "Phosphorus supports flowering and pod formation.",
        "Moderate rainfall prevents waterlogging and disease."
    ],
    "blackgram": [
        "Blackgram thrives in warm and moist conditions.",
        "Nitrogen and phosphorus improve flowering and pod development.",
        "Potassium enhances grain quality.",
        "Well-drained soil prevents root rot."
    ],
    "mungbean": [
        "Mungbean grows well in warm, sunny climates.",
        "Nitrogen and potassium promote pod and seed development.",
        "Moderate phosphorus supports flowering.",
        "Adequate rainfall ensures healthy plant growth."
    ],
    "mothbeans": [
        "Mothbeans tolerate hot, arid conditions.",
        "Nitrogen and potassium improve seed yield.",
        "Phosphorus supports root and flower development.",
        "Low rainfall is ideal; waterlogging must be avoided."
    ],
    "pigeonpeas": [
        "Pigeonpeas thrive in semi-arid regions.",
        "Nitrogen fixation enhances soil fertility.",
        "Phosphorus and potassium support flowering and pod formation.",
        "Moderate rainfall ensures proper fruiting."
    ],
    "kidneybeans": [
        "Kidneybeans prefer moderate temperatures and humidity.",
        "Nitrogen supports healthy leaf and pod growth.",
        "Phosphorus enhances root and flowering development.",
        "Adequate rainfall is essential for good yield."
    ],
    "chickpea": [
        "Chickpeas grow well in cool, dry climates.",
        "Nitrogen and potassium support flowering and grain filling.",
        "Phosphorus enhances root development.",
        "Low humidity reduces risk of fungal diseases."
    ],
    "coffee": [
        "Coffee grows well in moderate temperatures and high humidity.",
        "Balanced soil nutrients enhance bean flavor.",
        "Slightly acidic pH suits coffee root systems.",
        "Adequate rainfall supports flowering and fruiting cycles."
    ]
}


# ------------------------------------------------------------
# PREDICTION FUNCTION
# ------------------------------------------------------------
def predict_crop():
    try:
        N = float(entry_N.get())
        P = float(entry_P.get())
        K = float(entry_K.get())
        temperature = float(entry_temp.get())
        humidity = float(entry_hum.get())
        ph = float(entry_ph.get())
        rainfall = float(entry_rain.get())

        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        mx_features = mx.transform(features)
        sc_features = sc.transform(mx_features)
        prediction = model.predict(sc_features)[0]

        crop_name = crop_dict[prediction]
        market_price = price_dict[prediction]

        lbl_result.config(
            text=f"🌾 Recommended Crop: {crop_name.upper()}\n💰 Market Price: ₹{market_price}/kg",
            fg="green"
        )

        # ----- 🔍 SEARCH IMAGE -----
        image_folder = os.path.join("static", "images")
        valid_exts = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

        image_path = None
        for ext in valid_exts:
            possible_path = os.path.join(image_folder, f"{crop_name}{ext}")
            if os.path.exists(possible_path):
                image_path = possible_path
                break

        # ----- 🖼️ DISPLAY IMAGE -----
        if image_path:
            img = Image.open(image_path)
            img = img.resize((250, 250))
            img_tk = ImageTk.PhotoImage(img)
            lbl_image.config(image=img_tk, text="")
            lbl_image.image = img_tk
        else:
            lbl_image.config(image='', text="🚫 No image found for this crop", fg="red")

        # ----- 📜 DISPLAY EXPLANATION -----
        if crop_name in explanation_dict:
            reasons = explanation_dict[crop_name]
        else:
            reasons = [
                "This crop suits the current soil and weather pattern.",
                "Nutrient and pH levels are within optimal range.",
                "Rainfall and temperature support healthy growth.",
                "The model identified this crop as the best economic choice."
            ]

        explanation_text = "\n".join([f"• {r}" for r in reasons])
        lbl_explanation.config(text=explanation_text, fg="#333333")

    except Exception as e:
        messagebox.showerror("Error", f"Invalid Input or Missing File:\n{str(e)}")


# ------------------------------------------------------------
# GUI DESIGN
# ------------------------------------------------------------
root = tk.Tk()
root.title("🌱 Crop & Market Price Prediction System")
root.geometry("750x750")
root.configure(bg="#f1f1f1")

# Title Label
tk.Label(root, text="🌾 Crop Recommendation System", font=("Arial", 20, "bold"),
         bg="#f1f1f1", fg="green").pack(pady=10)

# Input Frame
frame = tk.Frame(root, bg="#f1f1f1")
frame.pack(pady=10)

labels = ["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)",
          "Temperature (°C)", "Humidity (%)", "pH", "Rainfall (mm)"]

entries = []
for i, lbl in enumerate(labels):
    tk.Label(frame, text=lbl, bg="#f1f1f1", font=("Arial", 12)).grid(row=i, column=0, padx=10, pady=5, sticky="e")
    entry = tk.Entry(frame, width=20, font=("Arial", 12))
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries.append(entry)

entry_N, entry_P, entry_K, entry_temp, entry_hum, entry_ph, entry_rain = entries

# Predict Button
tk.Button(root, text="🔍 Predict Crop", command=predict_crop, font=("Arial", 14, "bold"),
          bg="green", fg="white", relief="raised", padx=20, pady=5).pack(pady=15)

# Result Label
lbl_result = tk.Label(root, text="", bg="#f1f1f1", font=("Arial", 14, "bold"))
lbl_result.pack(pady=10)

# Image Display
lbl_image = tk.Label(root, bg="#f1f1f1")
lbl_image.pack(pady=10)

# Explanation Label
lbl_explanation = tk.Label(root, text="", bg="#f1f1f1", font=("Arial", 12), justify="left", wraplength=700)
lbl_explanation.pack(pady=10)

# Run App
root.mainloop()
