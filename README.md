# Industrial AI Vision Platform

Production-oriented Computer Vision system for Automated Casting Defect Detection using Deep Learning, Explainable AI, Robustness Engineering, and Drift Monitoring.

---

## Overview

Industrial AI Vision Platform is an end-to-end Machine Learning project that detects defects in casting products using Computer Vision.

The project goes beyond simple image classification and demonstrates real-world ML Engineering practices including:

* Transfer Learning
* Feature Embedding Extraction
* Explainable AI (Grad-CAM)
* Cross Validation
* Data Drift Simulation
* Drift Detection
* Robustness Engineering
* Early Stopping
* Model Versioning

The goal is to mimic challenges faced by production AI systems deployed in manufacturing environments.

---

## Business Problem

Manual inspection of casting products is:

* Slow
* Expensive
* Inconsistent
* Error-prone

Defective products may pass inspection while good products may be rejected.

This platform automates defect detection using Deep Learning and provides explainable predictions for quality assurance teams.

---

## Dataset

### Casting Product Image Dataset

Classes:

| Class             | Label |
| ----------------- | ----- |
| OK Product        | 0     |
| Defective Product | 1     |

Dataset Structure:

```text
data/
└── casting_data/
    └── casting_data/
        ├── train/
        │   ├── ok_front/
        │   └── def_front/
        │
        └── test/
            ├── ok_front/
            └── def_front/
```

Training Samples:

* Good Castings: 2875
* Defective Castings: 3758

Test Samples:

* Good Castings: 262
* Defective Castings: 453

---

## Project Architecture

```text
Industrial AI Vision Platform
│
├── Data Pipeline
│
├── Dataset Loader
│
├── Transfer Learning
│   └── ResNet18
│
├── Training Pipeline
│
├── Feature Embedding Extraction
│
├── Explainable AI
│   └── Grad-CAM
│
├── Cross Validation
│
├── Drift Simulation
│
├── Drift Detection
│
├── Robust Training
│
└── Evaluation
```

---

## Tech Stack

### Machine Learning

* PyTorch
* Torchvision
* NumPy
* Scikit-Learn

### Visualization

* Matplotlib
* OpenCV

### Development

* Python
* VS Code

---

## Implemented Features

### 1. Transfer Learning

Model:

* ResNet18 (ImageNet Pretrained)

Benefits:

* Faster convergence
* Better feature extraction
* Reduced training time

---

### 2. Feature Embedding Extraction

Deep features are extracted from the penultimate layer of ResNet18.

Output:

```text
Embedding Shape:
(715, 512)
```

Applications:

* Similarity Search
* Clustering
* Drift Detection
* Anomaly Detection

---

### 3. K-Fold Cross Validation

Implemented 5-Fold Cross Validation.

Results:

Accuracy:

```text
Mean = 99.58%
Std  = 0.34%
```

F1 Score:

```text
Mean = 99.67%
```

Observation:

Model generalizes consistently across folds.

---

### 4. Explainable AI (Grad-CAM)

Implemented custom Grad-CAM visualization.

Capabilities:

* Highlights defect regions
* Explains model decisions
* Improves trustworthiness

Example Output:

```text
Prediction: Defective
Confidence: 1.00

Heatmap correctly focused on defect region.
```

---

### 5. Data Drift Simulation

Simulated real-world production challenges:

* Brightness shifts
* Blur
* Noise
* Sharpness degradation

Purpose:

Evaluate model robustness under changing environmental conditions.

---

### 6. Embedding Drift Detection

Compared embeddings from:

* Original Dataset
* Drifted Dataset

Result:

```text
Embedding Drift Distance:
19.288208
```

Interpretation:

Severe feature distribution shift detected.

---

### 7. Robustness Engineering (V3 Pipeline)

Added:

* Stratified Train/Validation Split
* Probabilistic Data Augmentation
* Early Stopping
* Best Model Checkpointing

Augmentations:

* Horizontal Flip
* Rotation
* Blur
* Contrast Shift
* Brightness Shift
* Sharpness Variation

---

### 8. Early Stopping

Prevents overfitting by stopping training when validation performance stops improving.

Configuration:

```python
patience = 5
```

---

### 9. Model Checkpointing

Automatically saves the best-performing model.

Output:

```text
models/best_v3_model.pth
```

---

## Model Performance

### Clean Test Set

| Metric    | Value   |
| --------- | ------- |
| Accuracy  | 99.58%  |
| Precision | 100.00% |
| Recall    | 99.34%  |
| F1 Score  | 99.67%  |

Confusion Matrix:

```text
[[262   0]
 [  3 450]]
```

---

## Drifted Test Set (Baseline)

| Metric   | Value  |
| -------- | ------ |
| Accuracy | 47.27% |
| Recall   | 17.44% |
| F1 Score | 29.53% |

Observation:

Severe performance degradation under production drift.

---

## Drifted Test Set (Robust V3 Model)

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 60.56% |
| Precision | 75.52% |
| Recall    | 55.85% |
| F1 Score  | 64.21% |

Confusion Matrix:

```text
[[180  82]
 [200 253]]
```

Improvement:

* Recall improved from 17.44% → 55.85%
* Defect detection significantly more robust

---

## Key Learnings

This project demonstrates:

* Production ML Pipelines
* Transfer Learning
* Explainable AI
* Drift Monitoring
* Robustness Engineering
* Validation Strategies
* Model Generalization

The focus is not only achieving high accuracy but ensuring reliability under real-world deployment conditions.

---

## Future Improvements

* MLflow Experiment Tracking
* Model Registry
* Hyperparameter Optimization
* ONNX Export
* FastAPI Deployment
* Dockerization
* Kubernetes Deployment
* Real-Time Video Inspection
* Active Learning Pipeline

---

## Author

Vedant Dhavan

Aspiring Agentic AI & Machine Learning Engineer focused on:

* Computer Vision
* MLOps
* AI Systems
* Production Machine Learning
