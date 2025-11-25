import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

# ----------------------------------------------------------
# 1️⃣ Load Trained Model and Class Names
# ----------------------------------------------------------
MODEL_PATH = "plant_disease_model.h5"
CLASS_PATH = "class_names.pkl"

print("🔄 Loading model and class names...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "rb") as f:
    class_names = pickle.load(f)

print(f"✅ Model and class names loaded successfully!")
print(f"Detected classes: {class_names}\n")

# ----------------------------------------------------------
# 2️⃣ Select Single Image Using File Dialog
# ----------------------------------------------------------
root = tk.Tk()
root.withdraw()  # Hide main tkinter window

file_path = filedialog.askopenfilename(
    title="Select a Leaf Image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)

if not file_path:
    print("❌ No image selected. Exiting.")
    exit()

print(f"🖼️ Selected Image: {os.path.basename(file_path)}")

# ----------------------------------------------------------
# 3️⃣ Preprocess Image
# ----------------------------------------------------------
def preprocess_image(img_path, img_size=(224, 224)):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

img_tensor = preprocess_image(file_path)

# ----------------------------------------------------------
# 4️⃣ Make Prediction
# ----------------------------------------------------------
pred = model.predict(img_tensor, verbose=0)
class_index = np.argmax(pred)
predicted_class = class_names[class_index]
confidence = np.max(pred) * 100

print(f"\n✅ Prediction Result:")
print(f"Image Name     → {os.path.basename(file_path)}")
print(f"Predicted Class → {predicted_class}")


# ----------------------------------------------------------
# 5️⃣ Display Image with Prediction
# ----------------------------------------------------------
plt.figure(figsize=(5, 5))
img = image.load_img(file_path)
plt.imshow(img)
plt.axis("off")
plt.title(f"{predicted_class}", fontsize=12, color='green')
plt.show()
