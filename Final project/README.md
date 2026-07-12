# 🚗 Vehicle Movement State Detection

> A classical Machine Learning pipeline for classifying vehicle motion states from dashcam image sequences using handcrafted spatio-temporal features.

**Course:** DS203 – Programming for Data Science  
**Institute:** IIT Bombay  
**Guide:** Prof. Vinay Kulkarni  
**Duration:** March 2026 – May 2026

---

## 📌 Project Overview

This project focuses on identifying the motion state of a vehicle from sequential dashcam images using classical Computer Vision and Machine Learning techniques.

Unlike deep learning approaches, this project relies on carefully engineered spatio-temporal features extracted from image sequences to capture vehicle dynamics.

The final model classifies each frame into one of four vehicle motion states:

- 🟢 AHEAD
- 🔴 HALTED
- ↩️ SHARP-LEFT
- ↪️ SHARP-RIGHT

---

## 📊 Dataset

- **7,900+** annotated dashcam sequential images
- **4** vehicle motion classes
- Sequential image frames captured under diverse driving conditions

---

## 🔍 Feature Engineering

Designed **13+ handcrafted spatio-temporal features** to characterize vehicle movement, including:

- Optical Flow
- Sobel Gradient Features
- Frame Difference Statistics
- Motion Asymmetry
- Entropy
- Variance
- Curl
- Rolling-window temporal statistics

These features capture both spatial appearance and temporal motion information.

---

## ⚠️ Class Imbalance Handling

The dataset exhibited significant class imbalance.

To improve minority-class performance:

- Hybrid **SMOTE** oversampling
- Random Under Sampling
- Class-wise Threshold Optimization
- Stratified Cross Validation

These techniques substantially improved Macro F1-score and minority-class recall.

---

## 🤖 Models Evaluated

Five classical Machine Learning models were benchmarked:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Gradient Boosting
- Shallow Artificial Neural Network

---

## 📈 Model Selection

Gradient Boosting consistently produced the best balance across all motion classes.

Additional threshold optimization further improved predictions for underrepresented classes.

---

## 🏆 Final Performance

| Metric | Score |
|---------|-------|
| Accuracy | **95.3%** |
| Macro F1-Score | **0.79** |
| Cross Validation | **5-Fold Stratified CV** |
| Features Engineered | **13+** |
| Images Used | **7,900+** |

---

## 🛠️ Tech Stack

- Python
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Jupyter Notebook

---

## 📂 Repository Structure

```
Vehicle-Movement-State-Detection/
│
├── data/
├── notebooks/
├── models/
├── results/
├── images/
├── Vehicle_Movement_Classifier.ipynb
├── requirements.txt
└── README.md
```

---

## 📷 Sample Pipeline

```
Dashcam Frames
        │
        ▼
Image Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Class Imbalance Correction
        │
        ▼
Model Training
        │
        ▼
Threshold Optimization
        │
        ▼
Vehicle Motion Prediction
```

---

## 🚀 Key Contributions

- Developed a complete classical ML pipeline for vehicle motion classification.
- Engineered **13+** discriminative spatio-temporal features from sequential dashcam images.
- Corrected severe class imbalance using hybrid SMOTE and threshold tuning.
- Benchmarked **5** machine learning models using Stratified Cross Validation.
- Achieved **95.3% Accuracy** and **0.79 Macro F1-Score** using Gradient Boosting.

---

## 📖 Future Improvements

- Deep Learning (CNN + LSTM)
- Vision Transformers
- Real-time inference pipeline
- Object-aware vehicle motion estimation
- Multi-camera support

---

## 👨‍💻 Author

**Ajay Teja K**

Department of Civil Engineering  
Indian Institute of Technology Bombay
