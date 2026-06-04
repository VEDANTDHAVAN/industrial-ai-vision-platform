import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

X = np.load("outputs/embeddings.npy")
y = np.load("outputs/labels.npy")

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42
)

scores = cross_validate(
    model, X, y, cv=cv, scoring=["accuracy", "precision", "recall", "f1"],
    return_train_score=True
)

for metric in ["accuracy", "precision", "recall", "f1"]:
    test_scores = scores[f"test_{metric}"]

    print(f"\n{metric.upper()}")
    print("Scores:", test_scores)
    print("Mean:", test_scores.mean())
    print("Std:", test_scores.std())