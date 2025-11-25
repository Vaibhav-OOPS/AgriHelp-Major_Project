# ============================================================
# PLANT LEAF DISEASE DETECTION - CNN (VGG16 with local weights)
# ============================================================

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

# -------------------------------
# 1️⃣ Load Dataset
# -------------------------------
dataset_path = "dataset"  # 👈 Your dataset folder (10 subfolders inside)

img_size = (224, 224)
batch_size = 32

train_ds = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"✅ Classes detected: {class_names}")

# Normalize [0,1]
train_ds = train_ds.map(lambda x, y: (x / 255.0, y))
val_ds = val_ds.map(lambda x, y: (x / 255.0, y))

# Improve performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# -------------------------------
# 2️⃣ Load VGG16 Model Locally
# -------------------------------
local_vgg16_path = "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"

base_model = VGG16(
    weights=local_vgg16_path,  # 👈 Local file
    include_top=False,
    input_shape=img_size + (3,)
)
base_model.trainable = False  # Freeze convolutional layers

# -------------------------------
# 3️⃣ Build Custom Model
# -------------------------------
model = Sequential([
    base_model,
    Flatten(),
    BatchNormalization(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -------------------------------
# 4️⃣ Training Configuration
# -------------------------------
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
    ModelCheckpoint("best_plant_model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)
]

# -------------------------------
# 5️⃣ Train Model
# -------------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=callbacks,
    verbose=1
)

# -------------------------------
# 6️⃣ Save Model and Class Names
# -------------------------------
model.save("plant_disease_model.h5")
with open("class_names.pkl", "wb") as f:
    pickle.dump(class_names, f)

print("✅ Model and class names saved successfully!")

# -------------------------------
# 7️⃣ Plot Accuracy and Loss
# -------------------------------
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.legend()

plt.tight_layout()
plt.show()
