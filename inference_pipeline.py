import cv2
import numpy as np
from keras.models import load_model

# Load models
sleep_model = load_model('models/sleep_detection_model.h5')
age_model = load_model('models/age_prediction_model.h5')

# Load face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def process_image(image_path):
    img = cv2.imread(image_path)
    faces = face_cascade.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.1, 5)

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        resized = cv2.resize(face_img, (224, 224)) / 255.0
        input_img = np.expand_dims(resized, axis=0)

        sleep_pred = sleep_model.predict(input_img)
        sleep_class = np.argmax(sleep_pred)

        if sleep_class == 1:  # assuming 1 = sleeping_person
            print("Sleeping person detected.")
            age_pred = age_model.predict(input_img)
            age_class = np.argmax(age_pred)
            print(f"Predicted age class: {age_class}")
        else:
            print("No sleeping person detected.")

# Example:
process_image('test_image.jpg')
