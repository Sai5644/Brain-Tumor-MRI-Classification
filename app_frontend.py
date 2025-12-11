# app_frontend.py
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import os

# ---- Setup ----
MODEL_PATH = "outputs/brain_tumor_model.h5"
CLASS_TYPES = ['pituitary', 'notumor', 'meningioma', 'glioma']

model = tf.keras.models.load_model(MODEL_PATH)
IMAGE_SIZE = (150, 150)

# ---- GUI ----
root = tk.Tk()
root.title("🧠 Brain Tumor MRI Classifier")
root.geometry("600x600")
root.config(bg="#f0f4f8")

title = Label(root, text="Brain Tumor MRI Classification", font=("Helvetica", 18, "bold"), bg="#f0f4f8", fg="#333")
title.pack(pady=20)

img_label = Label(root, bg="#f0f4f8")
img_label.pack(pady=10)

result_label = Label(root, text="", font=("Helvetica", 16, "bold"), bg="#f0f4f8")
result_label.pack(pady=20)

def predict_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)
    return CLASS_TYPES[class_idx], confidence

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return
    
    # Display image
    img = Image.open(file_path)
    img = img.resize((250, 250))
    img_tk = ImageTk.PhotoImage(img)
    img_label.config(image=img_tk)
    img_label.image = img_tk

    # Predict
    predicted_class, confidence = predict_image(file_path)
    result_label.config(text=f"Prediction: {predicted_class.upper()}\nConfidence: {confidence*100:.2f}%", fg="green")

upload_btn = Button(root, text="📁 Upload MRI Image", command=upload_image, bg="#4CAF50", fg="white",
                    font=("Helvetica", 14), padx=10, pady=5)
upload_btn.pack(pady=20)

exit_btn = Button(root, text="❌ Exit", command=root.quit, bg="red", fg="white",
                  font=("Helvetica", 14), padx=10, pady=5)
exit_btn.pack(pady=10)

root.mainloop()
