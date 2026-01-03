# 🌿 Plant Disease Detection using Deep Learning

##  Problem Description

Plant diseases significantly reduce crop yield and quality, especially in regions where access to agricultural experts is limited. Farmers often rely on visual inspection, which is error-prone and subjective.

This project aims to **automatically detect plant diseases from leaf images** using a **deep learning convolutional neural network (CNN)**. The solution allows users to upload an image of a plant leaf via a web interface and instantly receive:

- The predicted disease class (human-readable)
- The confidence score of the prediction

An additional **“Unidentified” disease class** is included to reduce false positives by capturing out-of-distribution images.

**Target users:**  
- Farmers  
- Agricultural extension workers  
- Researchers and students  

---

##  Dataset

### Primary Dataset
- **PlantVillage Dataset**
- Source: https://www.kaggle.com/datasets/emmarex/plantdisease

The dataset contains labeled leaf images across multiple crop types and disease categories.

### Additional "Unidentified" Class
To improve real-world robustness, an **Unidentified** class was manually created using unrelated plant and non-plant images.

#### How to obtain it:
1. Download the provided folder from Google Drive: https://drive.google.com/drive/folders/1lDiChOIP1ldlje4PCw7Gaw5nj5Jsnr_c?usp=sharing
2. Extract and copy the folder into: data/PlantVillage/
   
---

## Demo Video 
This video contained in the google drive link shows the demonstration of how the app works 
https://drive.google.com/file/d/13Q3nVva__DmW9_JR7QKxGPPvx3-oKUSo/view?usp=sharing

---

## 🔍 Exploratory Data Analysis (EDA)

EDA was performed in `notebook.ipynb` and includes:

- Class distribution analysis
- Sample visualization per class
- Image resolution inspection
- RGB channel distribution
- Dataset imbalance observations

## 🧠 Model Training

### Architecture
- Backbone: **efficientnet_b0 (transfer learning)**
- Pretrained on ImageNet
- Final fully connected layer adapted to number of classes

### Training Strategy
Multiple experiments were conducted:
- With and without dropout
- Different learning rates
- Data augmentation vs no augmentation
- Different optimizer settings

### Hyperparameters Tuned
- Learning rate
- Dropout rate
- Number of epochs
- Batch size

### Training Metrics
#### Accuracy over epochs
![Accuracy Curve](assets/accuracy.png)

#### Loss over epochs
![Loss Curve](assets/loss.png)

---

## 📈 Model Evaluation

### Classification Report
![Classification Report](assets/classification_report.png)

### Confusion Matrix
(All values shown explicitly for clarity)

![Confusion Matrix](assets/confusion_matrix.png)

---

## Notebook

- File: `notebook.ipynb`
- Contents:
- Data loading and cleaning
- EDA and visualization
- Model experiments
- Hyperparameter tuning
- Final evaluation

---

##  Training Script

- File: `train.py`
- Responsibilities:
- Load dataset
- Train final model
- Save model checkpoint
- Save class names and metadata

Model is saved to: models/plant_disease_classifier.pth



##  Model Deployment

### Web Application
- Framework: **Flask**
- File: `predict.py`
- Features:
  - Image upload
  - Prediction with confidence score
  - Image preview
  - Clean class name formatting

-

## Containerization

A Dockerfile is provided to containerize the application.

### Build image
 --on bash
docker build -t plant-disease-app .
docker run -p 5000:5000 plant-disease-app


