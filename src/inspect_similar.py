import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

embeddings = np.load("outputs/embeddings.npy")

sim = cosine_similarity(
    embeddings[0].reshape(1, -1),
    embeddings[1].reshape(1, -1)
)

print(sim)