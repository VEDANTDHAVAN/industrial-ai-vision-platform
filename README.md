# Industrial AI Vision Platform

## Overview

Industrial AI Vision Platform is an end-to-end computer vision system for automated casting defect detection.

The platform includes:

* Deep Learning model training
* Explainable AI (Grad-CAM)
* Drift Detection
* Model Monitoring
* MLflow Experiment Tracking
* Model Registry
* FastAPI Inference API
* Prediction Logging
* Confidence Analysis
* Production Monitoring

---

# Project Architecture

Dataset → Training Pipeline → Model Registry → FastAPI Inference → Prediction Logging → Monitoring → Drift Detection → Retraining

---

# Dataset

Casting Product Image Dataset

Classes:

* OK
* Defective

Training Images: 5306

Test Images: 715

Image Size:

224 × 224

---

# Model Versions

## V1

Baseline ResNet18

Features:

* Transfer Learning
* Binary Classification

Outcome:

Strong baseline performance on clean test data.

---

## V2

Explainable AI

Implemented:

* Grad-CAM
* Defect Localization

Outcome:

Verified model focuses on defect regions instead of irrelevant image areas.

---

## V3

Monitoring Foundations

Implemented:

* Embedding Extraction
* Feature Monitoring
* Drift Detection

Outcome:

Successfully detected severe domain drift.

Embedding Drift Distance:

19.288

Status:

Severe Drift Detected

---

## V4

Production Pipeline

Implemented:

* MLflow Experiment Tracking
* MLflow Model Registry
* FastAPI Inference Service
* Prediction Logging
* Monitoring Dashboard
* Confidence Monitoring

Model Registry:

CastingDefectDetector

Registered Version:

Version 1

Inference Endpoint:

POST /predict

Prediction Logging:

logs/predictions.csv

---

## V5

Robustness Improvement

Implemented:

* Strong Data Augmentation
* Label Smoothing
* Cosine Annealing Learning Rate Scheduler
* Early Stopping

Goal:

Improve performance under domain drift.

---

# Performance Results

## Clean Test Dataset

Model:

V5

Accuracy:

99.72%

Precision:

100.00%

Recall:

99.56%

F1 Score:

99.78%

Confusion Matrix:

[[262, 0],
[2, 451]]

---

## Drifted Dataset

Model:

V5

Accuracy:

62.38%

Precision:

73.83%

Recall:

62.91%

F1 Score:

67.94%

Confusion Matrix:

[[161, 101],
[168, 285]]

---

# Confidence Analysis

## Original Dataset

Accuracy:

99.58%

Average Confidence (Correct):

0.9951

Average Confidence (Wrong):

0.6937

Observation:

Strong confidence separation.

---

## Drifted Dataset

Accuracy:

63.92%

Average Confidence (Correct):

0.6384

Average Confidence (Wrong):

0.6086

Observation:

Model becomes uncertain under domain drift.

---

# Monitoring System

Current Monitoring Metrics:

* Prediction Counts
* Average Confidence
* Manual Review Rate
* Confidence Drop Alerts
* Retraining Alerts

Sample Monitoring Output:

Average Confidence: 0.7297

Manual Review Rate: 100%

Alerts:

* Confidence Drop Detected
* Excessive Manual Reviews
* Retraining Recommended

---

# MLflow Tracking

Tracked Parameters:

* Learning Rate
* Batch Size
* Optimizer
* Epochs
* Dataset Sizes

Tracked Metrics:

* Train Loss
* Validation Loss
* Train Accuracy
* Validation Accuracy
* Best Validation Loss

Tracked Artifacts:

* Model Checkpoints
* Registered Models

---

# Explainability

Implemented:

Grad-CAM

Capabilities:

* Visual Defect Localization
* Prediction Validation
* Model Auditing

Outcome:

Heatmaps focus on actual defect regions.

---

# Drift Detection

Implemented:

Embedding-Based Drift Detection

Current Drift Distance:

19.288

Status:

Severe Drift Detected

Action:

Retraining Recommended

---

# Current Challenges

Primary Challenge:

Generalization to drifted real-world production images.

Current Gap:

Clean Accuracy:

99.72%

Drift Accuracy:

62.38%

Accuracy Gap:

37.34%

---

# Next Milestone (V6)

Objectives:

* Replace ResNet18 with EfficientNet-B0
* Improve drift robustness
* Increase drift accuracy beyond 75%
* Reduce manual review rate
* Improve confidence separation under drift

Target Metrics:

Clean Accuracy > 99%

Drift Accuracy > 75%

Manual Review Rate < 30%

Confidence Gap > 0.10

---

# Technology Stack

Python, PyTorch, TorchVision, 
FastAPI, MLflow, Pandas, NumPy
Scikit-Learn, OpenCV Matplotlib

---

# Project Status

Current Stage:

Production-Oriented Industrial Computer Vision System

Progress:

Approximately 80% complete toward a deployable MLOps-enabled defect detection platform.
