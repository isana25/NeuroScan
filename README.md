# 🧠 Brain Tumor Detection with CNN + Gradio

This project implements a deep learning model using Convolutional Neural Networks (CNN) to detect brain tumors from MRI scans. It features a clean, interactive Gradio interface that allows users to upload an image and receive a classification: Tumor or No Tumor.

---

## 📁 Dataset Used
**[Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)**  
- Binary classes: `yes` (tumor present) and `no` (no tumor)  
- Downloaded using `kagglehub` within the notebook

---

## 🏗️ Model Architecture
Custom CNN built with TensorFlow/Keras:

- Input: 128x128 RGB image  
- Layers:
  - Conv2D (32 filters) → MaxPooling  
  - Conv2D (64 filters) → MaxPooling  
  - Flatten → Dropout (0.5)  
  - Dense (64) → Dense (2 - softmax)  

---

## 🔍 What the Project Does
- Downloads and loads MRI dataset using `kagglehub`
- Preprocesses and normalizes images
- Trains a CNN model to classify scans as Tumor/No Tumor
- Provides an interactive Gradio interface for image-based prediction

---

## 🔧 Tech Stack
**Tech stack**: Python, TensorFlow/Keras, NumPy, OpenCV, Gradio, Google Colab, Kaggle API

---

## 🚀 Google Colab Notebook
**👉 [Try the Colab Notebook](https://colab.research.google.com/drive/1ZKml7FB6UvHkXaF8tQqXkJUexhXtXX65?usp=sharing)**

---

## 📌 Future Enhancements
- Add Grad-CAM for explainability  
- Include multi-class tumor classification  
- Deploy model as a Hugging Face Space

---

## 📜 License
MIT License (or your choice)
