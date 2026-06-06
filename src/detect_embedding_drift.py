import numpy as np
from scipy.spatial.distance import cdist

train_embeddings = np.load("outputs/embeddings.npy")

drift_embeddings = np.load("outputs/drift_embeddings.npy")

train_mean = train_embeddings.mean(axis=0)
drift_mean = drift_embeddings.mean(axis=0)

distance = np.linalg.norm(
    train_mean - drift_mean
)

print("\nEmbedding Drift Distance")
print(distance)

DRIFT_THRESHOLD = 10.0

if distance > DRIFT_THRESHOLD:
    print("Drift Detected!!")
else:
    print("No Drift")