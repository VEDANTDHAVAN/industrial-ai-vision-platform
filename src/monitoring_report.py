import json
from pathlib import Path
from datetime import datetime
import numpy as np

REPORT_PATH = Path("outputs/monitoring_report.json")

ORIGINAL_METRICS = {
    "accuracy": 0.9958,
    "precision": 1.0,
    "recall": 0.9934,
    "f1": 0.9967,
}

DRIFT_METRICS = {
    "accuracy": 0.4727,
    "precision": 0.9634,
    "recall": 0.1744,
    "f1": 0.2953,
}

DRIFT_THRESHOLD = 10.0

def main():
    train_embeddings = np.load("outputs/embeddings.npy")
    drift_embeddings = np.load("outputs/drift_embeddings.npy")

    train_mean = train_embeddings.mean(axis=0)
    drift_mean = drift_embeddings.mean(axis=0)

    drift_distance = float(np.linalg.norm(train_mean - drift_mean))
    drift_detected = drift_distance > DRIFT_THRESHOLD

    recall_drop = ORIGINAL_METRICS["recall"] - DRIFT_METRICS["recall"]
    f1_drop = ORIGINAL_METRICS["f1"] - DRIFT_METRICS["f1"]

    if drift_detected and recall_drop > 0.3:
        recommendation = "Severe drift detected. Retraining with augmented or new production images is recommended."
    elif drift_detected:
        recommendation = "Embedding drift detected. Monitor predictions and collect labeled production samples."
    else:
        recommendation = "No major drift detected."

    report = {
        "generated_at": datetime.now().isoformat(),
        "embedding_drift_distance": drift_distance,
        "drift_threshold": DRIFT_THRESHOLD,
        "drift_detected": drift_detected,
        "original_metrics": ORIGINAL_METRICS,
        "drift_metrics": DRIFT_METRICS,
        "recall_drop": recall_drop,
        "f1_drop": f1_drop,
        "recommendation": recommendation,
    }
    
    REPORT_PATH.parent.mkdir(exist_ok=True)

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()