 ##🧠 Methodology

1️⃣ Data Collection & Preparation

Collected labeled image data for:

Sleep detection (2 classes: sleeping_person, awake_person)

Age prediction (95 classes for ages 1–95)

Organized datasets into Keras-compatible folder structures

Rescaled all images to 224x224 and normalized pixel values (0–1)

2️⃣ Model Architecture

Used VGG16 (ImageNet pretrained) as frozen base for transfer learning

Sleep detection model:

VGG16 + GlobalAveragePooling2D + Dense(2, softmax)

Age prediction model:

VGG16 + GlobalAveragePooling2D + Dense(95, softmax)

3️⃣ Data Augmentation & Class Balancing

Augmented training data using:

Rotation, zoom, shifts, horizontal flip, brightness variation

Applied class weights (computed via sklearn) to handle imbalanced sleep/awake samples

4️⃣ Training & Evaluation

Trained each model using ImageDataGenerator

Sleep detection:

Epochs: 5, Metric: Accuracy, Confusion Matrix, Classification Report

Age prediction:

Epochs: 3, Metrics: Top-1 and Top-5 Accuracy

5️⃣ Face Detection & Inference

Detected faces with OpenCV Haar Cascade

For each face:

Predicted sleep status

If sleeping, passed to age prediction model

6️⃣ GUI Interaction

Implemented a Tkinter GUI to upload images

Displays results (number of sleeping persons + predicted ages)

Designed for real-time, multi-face input scenarios
