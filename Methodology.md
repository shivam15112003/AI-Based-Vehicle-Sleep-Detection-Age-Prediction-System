
## 🧠 Methodology

### 1️⃣ Data Collection & Preparation

- Collected labeled image datasets for two independent tasks:
  - **Sleep Detection**: 2 classes (`sleeping_person`, `awake_person`)
  - **Age Prediction**: 95 classes (ages 1–95)
- Organized datasets into Keras-compatible directory structures (`sleep_train/`, `age_train/`, etc).
- Rescaled all images to **224x224** pixels to match VGG16 input requirements.
- Normalized pixel values to a range of [0, 1].

---

### 2️⃣ Model Architecture

#### Sleep Detection Model (Binary Classification)
- Transfer learning with **VGG16** (pretrained on ImageNet)
- Used VGG16 as frozen feature extractor (`trainable = False`).
- Added custom classification head:
  - `GlobalAveragePooling2D`
  - `Dense(2, activation='softmax')`

#### Age Prediction Model (Multi-Class Classification)
- Transfer learning with **VGG16** (pretrained on ImageNet)
- Initial training with frozen VGG16
- Fine-tuned last few convolutional layers to improve performance on age classification.
- Custom classification head:
  - `GlobalAveragePooling2D`
  - `Dense(95, activation='softmax')`

---

### 3️⃣ Data Augmentation & Class Balancing

- Applied extensive data augmentation to improve model generalization:
  - Rotation (up to ±15–20 degrees)
  - Zoom (up to 20%)
  - Width & height shifts
  - Horizontal flipping
  - Brightness variation (0.7 – 1.3)
- Addressed class imbalance for sleep detection using **class weights** computed via `sklearn`.

---

### 4️⃣ Model Training & Evaluation

#### Sleep Detection Model

- Optimizer: Adam
- Loss: Categorical Crossentropy
- Epochs: 5
- Metrics: Accuracy, Confusion Matrix, Classification Report

#### Age Prediction Model

- Optimizer: Adam
- Loss: Categorical Crossentropy
- Epochs: 3 (initial frozen training)
- Fine-tuned last layers with smaller learning rate for additional 5 epochs
- Metrics: Top-1 Accuracy, Top-5 Accuracy

---

### 5️⃣ Face Detection & Inference Pipeline

- Used **OpenCV Haar Cascade** to detect faces in uploaded images.
- For each detected face:
  - Applied sleep detection model.
  - If classified as `sleeping_person`, passed the cropped face image to the age prediction model.

---

### 6️⃣ GUI Interaction

- Built an interactive **Tkinter GUI** for real-time analysis.
- User uploads images through GUI.
- Displays:
  - Total number of faces detected.
  - Count of sleeping persons.
  - Predicted age(s) for sleeping individuals.
- Supports multiple faces per image.

---
