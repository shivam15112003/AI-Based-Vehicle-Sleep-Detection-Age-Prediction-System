## 🚗 AI-Based Vehicle Sleep Detection & Age Prediction System

This project is a complete **deep learning + computer vision system** that detects whether individuals inside a vehicle are sleeping and predicts their age from images. The system uses **two specialized CNN models (VGG16-based)** with OpenCV for face detection and a simple Tkinter GUI for user interaction.

---

## 📌 Key Features

- 🧠 **Two Independent Specialized Models:**
  - **Sleep Detection Model**: Binary classification (sleeping vs awake) using VGG16 as feature extractor.
  - **Age Prediction Model**: Multi-class age classification (1–95 years) using fine-tuned VGG16.
  
- 🖼️ **Multi-Person Detection:**
  - Detects multiple faces per image using OpenCV Haar Cascade.
  - Applies sleep detection and age prediction independently on each detected face.

- 🔄 **Advanced Data Augmentation:**
  - Rotation, zoom, shift, horizontal flip, brightness adjustments to improve generalization.

- ⚖️ **Class Balancing:**
  - Sleep detection uses class weighting to handle dataset imbalance.

- 📊 **Performance Metrics:**
  - Sleep Detection Accuracy: **~97%**
  - Age Prediction Top-1 Accuracy: **~80-90% (expected based on dataset)**
  - Age Prediction Top-5 Accuracy: **~85-95% (expected)**

---

## 🔧 Technologies Used

- **Python 3.x**
- **TensorFlow / Keras** (VGG16-based CNNs)
- **OpenCV** (Haar Cascade Face Detection)
- **Tkinter** (Graphical User Interface)
- **Matplotlib, Seaborn** (Visualization)
- **scikit-learn** (Class weights, evaluation metrics)

---

## 📁 Project Directory Structure

```
project/
├── haarcascade_frontalface_default.xml  # OpenCV face detector
├── datasets/
│   ├── sleep_train/
│   ├── sleep_test/
│   ├── age_train/
│   └── age_test/
├── models/
│   ├── sleep_detection_model.h5
│   └── age_prediction_model.h5
├── sleep_detection.py        # Train sleep detection model
├── age_prediction.py         # Train age prediction model
├── inference_pipeline.py     # Inference pipeline (combined)
├── sleep_detection_gui.py    # GUI interface
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Prepare Dataset

Organize your dataset according to the folder structure above.  
Each subfolder must contain class-wise image folders (required by Keras `flow_from_directory()`).

### 3️⃣ Train Models

- **Sleep Detection Model:**

```bash
python sleep_detection.py
```

- **Age Prediction Model:**

```bash
python age_prediction.py
```

*This will generate two trained models stored in `models/`.*

### 4️⃣ Run Inference via GUI

```bash
python sleep_detection_gui.py
```

Upload any image through the GUI and see both sleep status and predicted age for each detected person.

---

## 🧪 Model Evaluation Summary

### Sleep Detection (Binary Classification)

- Classes: `awake_person`, `sleeping_person`
- Model: VGG16 (frozen feature extractor + classification head)
- Metrics: Confusion matrix, classification report
- Accuracy: **~97%**

### Age Prediction (Multi-Class Classification)

- Classes: 95 age classes (1–95 years)
- Model: VGG16 (fine-tuned on last layers)
- Metrics: Top-1 and Top-5 accuracy
- Expected Top-1 Accuracy: **~80-90%**  
- Expected Top-5 Accuracy: **~90-95%**

> *Exact accuracy may vary depending on dataset size, quality, and balance.*

---

## 📈 Potential Future Improvements

- Real-time video stream integration for continuous monitoring.
- Age binning (e.g., 0–10, 11–20 years) for more robust predictions.
- Lightweight models (MobileNet, EfficientNet) for edge deployment.
- Add result logging and exporting capabilities.
- Integration with alert systems for live vehicle monitoring.

---

## 📷 Sample Output

```
Detected: 3 faces
Sleeping Persons: 2
Predicted Ages: 24, 35
```

---

## 👨‍💻 Author

**Shivam Sharma**  
[GitHub Profile](https://github.com/shivam15112003)
