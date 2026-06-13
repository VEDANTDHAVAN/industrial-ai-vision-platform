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

### Version 4 — ResNet18 + Augmentation + Early Stopping

#### Clean Test Dataset

| Metric    | Value   |
| --------- | ------- |
| Accuracy  | 99.72%  |
| Precision | 100.00% |
| Recall    | 99.56%  |
| F1 Score  | 99.78%  |

Confusion Matrix:

```text
[[262   0]
 [  2 451]]
```

#### Drifted Dataset

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

---

### Version 5 — Improved ResNet18

Improvements:

* Better augmentations
* Improved train/validation split
* MLflow experiment tracking
* Model Registry integration

#### Drifted Dataset

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 62.38% |
| Precision | 73.83% |
| Recall    | 62.91% |
| F1 Score  | 67.94% |

Confusion Matrix:

```text
[[161 101]
 [168 285]]
```

Result:

* Improved drift robustness over V4.
* Still struggled under severe domain shift.

---

### Version 6 — EfficientNet-B0

Objective:

* Improve feature extraction capability.
* Increase robustness to unseen image distributions.

#### Clean Test Dataset

| Metric    | Value   |
| --------- | ------- |
| Accuracy  | 99.72%  |
| Precision | 100.00% |
| Recall    | 99.56%  |
| F1 Score  | 99.78%  |

Confusion Matrix:

```text
[[262   0]
 [  2 451]]
```

#### Drifted Dataset

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 62.94% |
| Precision | 63.95% |
| Recall    | 95.14% |
| F1 Score  | 76.49% |

Confusion Matrix:

```text
[[ 19 243]
 [ 22 431]]
```

Observations:

* Highest defect recall achieved.
* Very aggressive defect prediction behavior.
* Large number of false positives.
* Best overall defect detector under drift.

#### Confidence Analysis

Accuracy: 62.94%

Average Confidence (Correct): 0.9850

Average Confidence (Wrong): 0.9909

Observation:

The model was highly overconfident and often more confident when wrong than when correct.

---

### Version 7 — EfficientNet-B0 + Focal Loss

Objective:

* Improve probability calibration.
* Reduce overconfidence under drift.

#### Clean Test Dataset

| Metric    | Value   |
| --------- | ------- |
| Accuracy  | 99.72%  |
| Precision | 100.00% |
| Recall    | 99.56%  |
| F1 Score  | 99.78%  |

Confusion Matrix:

```text
[[262   0]
 [  2 451]]
```

#### Drifted Dataset

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 58.60% |
| Precision | 62.68% |
| Recall    | 85.65% |
| F1 Score  | 72.39% |

Confusion Matrix:

```text
[[ 31 231]
 [ 65 388]]
```

#### Confidence Analysis

Accuracy: 58.60%

Average Confidence (Correct): 0.8311

Average Confidence (Wrong): 0.8198

Observations:

* Calibration improved significantly.
* Correct predictions became more confident than incorrect predictions.
* Reduced model overconfidence.
* Defect detection performance decreased compared to V6.

Conclusion:

Focal Loss improved calibration but reduced defect recall and overall drift performance.

---

## Current Production Candidate

Model:

```text
models/best_v6_efficientnet.pth
```

Reason:

* Highest drift robustness achieved so far.
* Highest defect recall (95.14%).
* Lowest defect escape rate.
* Most suitable model for industrial quality inspection.

---

## Monitoring System

Implemented Features:

* Prediction logging to CSV
* Confidence monitoring
* Manual review tracking
* Drift performance analysis
* MLflow experiment tracking
* MLflow Model Registry
* FastAPI inference service
* Production-ready prediction endpoint

Example Monitoring Report:

```text
=== MONITORING REPORT ===

Total Predictions: 20

Average Confidence: 0.7297

Manual Review Rate: 100%

ALERT: Confidence drop detected

ALERT: Excessive manual reviews

RETRAINING RECOMMENDED
```

---

## Lessons Learned

1. Clean dataset accuracy above 99% does not guarantee robustness under distribution shift.

2. EfficientNet-B0 improved defect detection substantially under drift.

3. Focal Loss improved calibration but reduced industrial inspection performance.

4. Domain shift remains the primary challenge.

5. Monitoring and retraining pipelines are essential for production AI systems.

---

Current Status: Phase 6 Complete ✅

Completed:
- Dataset Pipeline
- Training Pipeline
- Evaluation Pipeline
- Drift Detection
- Confidence Analysis
- MLflow Tracking
- Model Registry
- FastAPI Inference API
- Prediction Logging
- Monitoring Dashboard

Current Best Model:
best_v6_efficientnet.pth

## Next Milestone (V8)

Objective:

Improve robustness against unseen visual distributions.

Planned Techniques:

* MixUp Augmentation
* Confidence Calibration
* Threshold Optimization
* Additional Drift Testing

Target Metrics:

* Drift Accuracy > 70%
* Recall > 90%
* Reduced False Positives
* Improved Production Stability

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
