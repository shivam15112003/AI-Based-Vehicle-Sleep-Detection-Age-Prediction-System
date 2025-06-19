import tensorflow as tf
import numpy as np
import cv2
import os
import tkinter as tk
from tkinter import filedialog
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dense
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from keras.metrics import TopKCategoricalAccuracy

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load base VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Build sleeping detection model
sleep_model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(2, activation='softmax')
])
sleep_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Build age prediction model
age_model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(95, activation='softmax')
])
age_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', TopKCategoricalAccuracy(k=5)])

# Load training and test data
train_datagen = ImageDataGenerator(rescale=1./255)
sleep_train_gen = train_datagen.flow_from_directory('datasets/sleeping_train', target_size=(224, 224), batch_size=32, class_mode='categorical')
sleep_test_gen = train_datagen.flow_from_directory('datasets/sleeping_test', target_size=(224, 224), batch_size=32, class_mode='categorical', shuffle=False)

# Train sleep detection model
sleep_model.fit(sleep_train_gen, epochs=5)

# Evaluate sleep model
sleep_eval = sleep_model.evaluate(sleep_test_gen)
print("Sleep Model Test Accuracy:", sleep_eval[1])

# Generate predictions for metrics
true_labels = sleep_test_gen.classes
predictions = sleep_model.predict(sleep_test_gen)
predicted_labels = np.argmax(predictions, axis=1)

# Confusion Matrix and Classification Report
cm = confusion_matrix(true_labels, predicted_labels)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=sleep_test_gen.class_indices, yticklabels=sleep_test_gen.class_indices)
plt.title('Confusion Matrix - Sleep Detection')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

print(classification_report(true_labels, predicted_labels, target_names=sleep_test_gen.class_indices.keys()))

# Load and train age prediction model
age_train_gen = train_datagen.flow_from_directory('datasets/age_train', target_size=(224, 224), batch_size=32, class_mode='categorical')
age_test_gen = train_datagen.flow_from_directory('datasets/age_test', target_size=(224, 224), batch_size=32, class_mode='categorical')

age_model.fit(age_train_gen, epochs=3)
age_eval = age_model.evaluate(age_test_gen)
print("Age Model Top-1 Accuracy:", age_eval[1])
print("Age Model Top-5 Accuracy:", age_eval[2])

# Helper to extract class name
def get_class_name(prediction, generator):
    idx = np.argmax(prediction)
    class_names = list(generator.class_indices.keys())
    return class_names[idx]

# Detect and classify image
def process_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    sleeping_count = 0
    age_predictions = []

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        face_resized = cv2.resize(face_img, (224, 224)) / 255.0
        face_exp = np.expand_dims(face_resized, axis=0)

        sleep_pred = sleep_model.predict(face_exp)
        label = get_class_name(sleep_pred, sleep_train_gen)

        if label == 'sleeping_person':
            sleeping_count += 1
            age_pred = age_model.predict(face_exp)
            age_label = get_class_name(age_pred, age_train_gen)
            age_predictions.append(age_label)

    return sleeping_count, age_predictions

# GUI setup
def browse_file():
    filename = filedialog.askopenfilename()
    img = cv2.imread(filename)
    if img is not None:
        count, ages = process_image(img)
        result = f"Found {count} sleeping person(s). Ages: {', '.join(ages)}" if count > 0 else "No sleeping person detected."
        result_label.config(text=result)
    else:
        result_label.config(text="Failed to load image.")

window = tk.Tk()
window.title("Vehicle Sleep Detection System")
window.geometry("500x300")

btn = tk.Button(window, text="Upload Image/Video", command=browse_file)
btn.pack(pady=10)

result_label = tk.Label(window, text="", wraplength=400)
result_label.pack(pady=20)

window.mainloop()
