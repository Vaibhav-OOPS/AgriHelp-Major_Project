# -------------------------------
# Imports
# -------------------------------
import pandas as pd
import pickle
import os
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# -------------------------------
# Load model and mappings
# -------------------------------
with open('rf_fertilizer_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('soil_mapping.pkl', 'rb') as f:
    soil_mapping = pickle.load(f)

with open('crop_mapping.pkl', 'rb') as f:
    crop_mapping = pickle.load(f)

with open('fertilizer_mapping.pkl', 'rb') as f:
    fertilizer_reverse_mapping = pickle.load(f)

# -------------------------------
# Fertilizer explanations
# -------------------------------
fertilizer_explanations = {
    "Urea": [
        "Urea provides high nitrogen content to boost leaf growth.",
        "It is suitable for crops with nitrogen-deficient soil.",
        "Supports overall vegetative development and yield."
    ],
    "Urea and DAP": [
        "Combination supplies both nitrogen and phosphorus.",
        "Enhances root development and early growth.",
        "Balanced nutrients improve overall crop performance."
    ],
    "Urea and MOP": [
        "Supplies nitrogen and potassium simultaneously.",
        "Promotes leaf growth and strong stem development.",
        "Ideal for crops needing potassium for fruiting."
    ],
    "DAP": [
        "DAP provides nitrogen and phosphorus for root & leaf growth.",
        "Suitable for phosphorus-deficient soils.",
        "Encourages early maturity and higher yield."
    ],
    "DAP and MOP": [
        "Supplies phosphorus and potassium for strong roots & fruits.",
        "Improves flowering and fruit quality.",
        "Balances nutrients for optimal crop health."
    ],
    "MOP": [
        "MOP supplies potassium for fruiting, flowering, and overall plant health.",
        "Strengthens stems and increases disease resistance.",
        "Ideal for potassium-deficient soils."
    ],
    "Good NPK": [
        "Balanced NPK fertilizer provides all essential nutrients.",
        "Supports growth, root development, and fruiting equally.",
        "Suitable for general purpose crop nutrition."
    ]
}

# -------------------------------
# Helper function to encode inputs safely
# -------------------------------
def encode_value(value, mapping):
    return mapping.get(value, -1)  # -1 if unseen

# -------------------------------
# Prediction function
# -------------------------------
def predict_fertilizer(input_dict):
    df = pd.DataFrame({
        'Temperature': [input_dict['Temperature']],
        'Humidity': [input_dict['Humidity']],
        'Rainfall': [input_dict['Rainfall']],
        'pH': [input_dict['pH']],
        'N': [input_dict['N']],
        'P': [input_dict['P']],
        'K': [input_dict['K']],
        'Soil': [encode_value(input_dict['Soil'], soil_mapping)],
        'Crop': [encode_value(input_dict['Crop'], crop_mapping)]
    })
    pred_idx = rf_model.predict(df)[0]
    return fertilizer_reverse_mapping[pred_idx]

# -------------------------------
# GUI Function
# -------------------------------
def on_predict():
    try:
        user_input = {
            'Temperature': float(entry_temp.get()),
            'Humidity': float(entry_hum.get()),
            'Rainfall': float(entry_rain.get()),
            'pH': float(entry_ph.get()),
            'N': float(entry_N.get()),
            'P': float(entry_P.get()),
            'K': float(entry_K.get()),
            'Soil': entry_soil.get(),
            'Crop': entry_crop.get()
        }

        fertilizer = predict_fertilizer(user_input)
        lbl_result.config(text=f"Recommended Fertilizer: {fertilizer}", fg="green")

        # Show explanation
        explanation_text = "\n".join(fertilizer_explanations.get(fertilizer, ["No explanation available."]))
        lbl_explanation.config(text=explanation_text, fg="blue")

        # Search for image
        image_folder = "fertilizer_images"  # 👈 Folder with fertilizer images
        valid_exts = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

        image_path = None
        for ext in valid_exts:
            possible_path = os.path.join(image_folder, f"{fertilizer}{ext}")
            if os.path.exists(possible_path):
                image_path = possible_path
                break

        # Display image
        if image_path:
            img = Image.open(image_path)
            img = img.resize((250, 250))
            img_tk = ImageTk.PhotoImage(img)
            lbl_image.config(image=img_tk)
            lbl_image.image = img_tk
        else:
            lbl_image.config(image='', text="🚫 Image not found", fg="red")

    except ValueError:
        messagebox.showerror("Error", "Please enter valid numeric values.")

# -------------------------------
# GUI Layout
# -------------------------------
root = tk.Tk()
root.title("Fertilizer Recommendation")
root.geometry("600x700")
root.configure(bg="#f0f0f0")

labels = ["Temperature", "Humidity", "Rainfall", "pH", "N", "P", "K", "Soil", "Crop"]
entries = []

frame = tk.Frame(root, bg="#f0f0f0")
frame.pack(pady=20)

for i, lbl in enumerate(labels):
    tk.Label(frame, text=lbl+":", font=("Arial", 12), bg="#f0f0f0").grid(row=i, column=0, sticky="e", padx=5, pady=5)
    entry = tk.Entry(frame, font=("Arial", 12))
    entry.grid(row=i, column=1, padx=5, pady=5)
    entries.append(entry)

entry_temp, entry_hum, entry_rain, entry_ph, entry_N, entry_P, entry_K, entry_soil, entry_crop = entries

# Predict button
tk.Button(root, text="Predict Fertilizer", font=("Arial", 14), bg="green", fg="white",
          command=on_predict).pack(pady=15)

# Result label
lbl_result = tk.Label(root, text="", font=("Arial", 14, "bold"), bg="#f0f0f0")
lbl_result.pack(pady=10)

# Explanation label
lbl_explanation = tk.Label(root, text="", font=("Arial", 12), bg="#f0f0f0", justify="left", wraplength=550)
lbl_explanation.pack(pady=10)

# Image display
lbl_image = tk.Label(root, bg="#f0f0f0")
lbl_image.pack(pady=10)

root.mainloop()
