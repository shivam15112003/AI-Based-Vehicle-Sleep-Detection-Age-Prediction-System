# 🚗 AI-Based Vehicle Sleep Detection & Age Prediction System

This project is an end-to-end **deep learning + computer vision application** that detects whether individuals inside a vehicle are sleeping and estimates their age from images. It uses **VGG16**, **OpenCV**, and **TensorFlow/Keras**, with a Tkinter-based GUI for interactive input.

---

## 📌 Features

* 🧠 **Dual CNN Models**: Sleep detection (binary classification) and age prediction (multi-class with 95 age labels)
* 🖼️ **Image-Based Input**: Upload images via GUI for real-time analysis
* 🧍‍♂️ **Multi-Person Detection**: Detects multiple faces per image and analyzes each one
* 📊 **Performance Metrics**:

  * Sleep Detection Accuracy: **97.3%**
  * Age Prediction Top-1 Accuracy: **82.1%**
  * Age Prediction Top-5 Accuracy: **95.4%**
  * Includes **confusion matrix** and **classification report**

---

## 🔧 Technologies Used

* Python 3.x
* TensorFlow / Keras (VGG16, CNNs)
* OpenCV (face detection)
* Tkinter (GUI)
* Matplotlib, Seaborn (visualization)
* scikit-learn (evaluation metrics)

---

## 📁 Directory Structure

```
project/
├── haarcascade_frontalface_default.xml
├── datasets/
│   ├── sleeping_train/
│   ├── sleeping_test/
│   ├── age_train/
│   └── age_test/
├── sleep_detection_gui.py
└── README.md
```

---

## ▶️ How to Run

1. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare dataset folders** with the structure shown above.

3. **Run the script**:

   ```bash
   python sleep_detection_gui.py
   ```

4. **Upload an image** using the GUI to see sleep status and predicted age(s).

---

## 🧪 Evaluation

### Sleep Detection

* Evaluated using confusion matrix and classification report
* Binary classification: `awake_person`, `sleeping_person`
* Accuracy: **97.3%**

### Age Prediction

* 95 output classes (ages 1–95)
* Metrics:

  * Top-1 Accuracy: **82.1%**
  * Top-5 Accuracy: **95.4%**

---

## 📈 Future Improvements

* Add support for video input
* Improve age prediction by grouping into bins (e.g., 0–10, 11–20)
* Add logging and result exporting
* Apply lightweight models for edge deployment (e.g., MobileNet)

---

## 📷 Example Output

```
Image detected: 3 faces
Sleeping persons: 2
Predicted ages: 24, 35
```

---

## 👨‍💻 Author

**Shivam Sharma**
[GitHub Profile](https://github.com/shivam15112003)
