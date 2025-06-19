# Enhanced version of sleep_detection_gui.py with improved preprocessing and class balancing

import tensorflow as tf
import numpy as np
import cv2
import os
import tkinter as tk
from tkinter import filedialog
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.metrics import TopKCategoricalAccuracy
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define base model for transfer learning
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

def build_model(num_classes):
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Sleep Detection Model
sleep_model = build_model(2)
sleep_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Age Prediction Model
age_model = build_model(95)
age_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', TopKCategoricalAccuracy(k=5)])

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load Datasets
sleep_train_gen = train_datagen.flow_from_directory('datasets/sleeping_train', target_size=(224, 224), batch_size=32, class_mode='categorical')
sleep_test_gen = test_datagen.flow_from_directory('datasets/sleeping_test', target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False)
age_train_gen = train_datagen.flow_from_directory('datasets/age_train', target_size=(224, 224), batch_size=32, class_mode='categorical')
age_test_gen = test_datagen.flow_from_directory('datasets/age_test', target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False)

# Class Weights for sleep detection
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(sleep_train_gen.classes), y=sleep_train_gen.classes)
class_weights = dict(enumerate(class_weights))

# Train Models
sleep_model.fit(sleep_train_gen, epochs=5, validation_data=sleep_test_gen, class_weight=class_weights)
age_model.fit(age_train_gen, epochs=3, validation_data=age_test_gen)

# Evaluate and Print Metrics
sleep_preds = np.argmax(sleep_model.predict(sleep_test_gen), axis=1)
sleep_true = sleep_test_gen.classes
print(classification_report(sleep_true, sleep_preds))
cm = confusion_matrix(sleep_true, sleep_preds)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix - Sleep Detection")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

age_eval = age_model.evaluate(age_test_gen)
print("Age Prediction Top-1 Accuracy:", age_eval[1])
print("Age Prediction Top-5 Accuracy:", age_eval[2])

# Prediction Pipeline

def get_class_name(pred, generator):
    idx = np.argmax(pred)
    return list(generator.class_indices.keys())[idx]

def process_image(img):
    faces = face_cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.1, 5)
    sleeping_count = 0
    age_predictions = []
    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        resized = cv2.resize(face_img, (224, 224)) / 255.0
        input_img = np.expand_dims(resized, axis=0)

        sleep_pred = sleep_model.predict(input_img)
        sleep_label = get_class_name(sleep_pred, sleep_train_gen)

        if sleep_label == 'sleeping_person':
            sleeping_count += 1
            age_pred = age_model.predict(input_img)
            age_label = get_class_name(age_pred, age_train_gen)
            age_predictions.append(age_label)

    return sleeping_count, age_predictions

# GUI Setup
window = tk.Tk()
window.title("Vehicle Sleep Detection System")
window.geometry("500x300")

result_label = tk.Label(window, text="", wraplength=400)
result_label.pack(pady=20)

def browse_file():
    filename = filedialog.askopenfilename()
    img = cv2.imread(filename)
    if img is not None:
        count, ages = process_image(img)
        if count > 0:
            result = f"Detected {count} sleeping person(s). Ages: {', '.join(ages)}"
        else:
            result = "No sleeping person detected."
    else:
        result = "Invalid image file."
    result_label.config(text=result)

upload_btn = tk.Button(window, text="Upload Image", command=browse_file)
upload_btn.pack(pady=10)

window.mainloop()
