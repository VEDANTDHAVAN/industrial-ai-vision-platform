# Industrial AI Vision Platform

An end-to-end Computer Vision and MLOps project for industrial defect detection, explainability, embedding analysis, and production drift monitoring.

## Overview

Industrial quality inspection systems often achieve excellent offline metrics but fail after deployment due to changing production environments such as lighting conditions, camera differences, image quality degradation, and manufacturing process changes.

This project explores the complete machine learning lifecycle from model training to explainability and drift analysis, simulating real-world challenges faced by production ML systems.

## Objectives

* Detect defective casting products using deep learning
* Learn transferable visual representations using CNN embeddings
* Visualize learned feature spaces
* Explain model decisions using Grad-CAM
* Simulate production data drift
* Evaluate model robustness under changing conditions
* Build foundations for production ML monitoring systems

---

## Dataset

### Casting Product Image Dataset

Binary Classification:

* OK Products
* Defective Products

Dataset Statistics:

| Split |   OK | Defective |
| ----- | ---: | --------: |
| Train | 2875 |      3758 |
| Test  |  262 |       453 |

Total Images:

* OK: 3137
* Defective: 4211
* Total: 7348

---

## Project Architecture

Image
↓
Preprocessing
↓
ResNet18 Transfer Learning
↓
Feature Embeddings (512-D)
↓
Classification
↓
Explainability (Grad-CAM)
↓
Embedding Analysis
↓
Drift Monitoring

---

## Technologies Used

### Deep Learning

* PyTorch
* TorchVision
* ResNet18 Transfer Learning

### Data Processing

* NumPy
* PIL
* OpenCV

### Evaluation

* Scikit-learn
* Cross Validation
* Classification Metrics

### Visualization

* Matplotlib
* PCA
* t-SNE

### Explainability

* Grad-CAM

---

## Features Implemented

### 1. Transfer Learning

Implemented a ResNet18-based defect detection model using pretrained ImageNet features.

### 2. Binary Defect Classification

Classes:

* OK
* Defective

### 3. Cross Validation

Performed 5-Fold Stratified Cross Validation on CNN embeddings.

Results:

| Metric    |    Mean |   Std |
| --------- | ------: | ----: |
| Accuracy  |  99.58% | 0.34% |
| Precision | 100.00% | 0.00% |
| Recall    |  99.34% | 0.54% |
| F1        |  99.67% | 0.27% |

### 4. CNN Embedding Extraction

Generated 512-dimensional embeddings from ResNet18 feature representations.

Embedding Shape:

(715, 512)

Applications:

* Similarity Search
* Vector Databases
* Retrieval Systems
* Representation Learning

### 5. Embedding Visualization

Implemented:

* PCA
* t-SNE

Observations:

* Clear separation between OK and Defective products
* Strong representation learning achieved by the CNN

### 6. Explainable AI (Grad-CAM)

Implemented Grad-CAM to visualize regions influencing predictions.

Results:

* Correct localization of defect regions
* High confidence predictions
* Improved model interpretability

### 7. Production Drift Simulation

Simulated real-world production drift using:

* Brightness changes
* Gaussian blur
* Image noise

This mimics:

* Camera changes
* Lighting variations
* Sensor degradation

---

## Baseline Evaluation

### Original Dataset

Accuracy: 99.58%

Precision: 100.00%

Recall: 99.34%

F1 Score: 99.67%

Confusion Matrix:

[[262, 0],
[3, 450]]

---

## Drift Evaluation

### Drifted Dataset

Accuracy: 47.27%

Precision: 96.34%

Recall: 17.44%

F1 Score: 29.53%

Confusion Matrix:

[[259, 3],
[374, 79]]

---

## Key Finding

Although the model achieved nearly perfect offline performance, simulated production drift caused recall to collapse from 99.34% to 17.44%.

This demonstrates a critical machine learning engineering lesson:

High test accuracy does not guarantee production robustness.

The project highlights the importance of:

* Data Drift Detection
* Model Monitoring
* Continuous Evaluation
* Retraining Pipelines

---

## Machine Learning Concepts Covered

* Transfer Learning
* Convolutional Neural Networks (CNNs)
* Representation Learning
* Embeddings
* Cross Validation
* Precision / Recall Analysis
* Confusion Matrix Interpretation
* Explainable AI (Grad-CAM)
* Data Drift
* Distribution Shift
* Production Monitoring

---

## Future Improvements

* Drift Detection Dashboard
* Evidently AI Integration
* Automated Retraining Pipeline
* MLflow Experiment Tracking
* Real-Time Inference API
* Docker Deployment
* Streamlit Monitoring Dashboard
* Object Detection using YOLO
* Predictive Maintenance Module

---

## Learning Outcomes

This project demonstrates not only defect classification but also the challenges of deploying machine learning systems in production environments, including explainability, representation learning, model robustness, and drift monitoring.
