# 🌿 Plant Disease Detection using Deep Learning

## 📖 Problem Description

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

## 🗂️ Dataset

### Primary Dataset
- **PlantVillage Dataset**
- Source: https://www.kaggle.com/datasets/emmarex/plantdisease

The dataset contains labeled leaf images across multiple crop types and disease categories.

### Additional "Unidentified" Class
To improve real-world robustness, an **Unidentified** class was manually created using unrelated plant and non-plant images.

#### How to obtain it:
1. Download the provided folder from Google Drive:
