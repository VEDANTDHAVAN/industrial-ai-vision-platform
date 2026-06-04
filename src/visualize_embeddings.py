import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

EMBEDDINGS_PATH = "outputs/embeddings.npy"
LABELS_PATH = "outputs/labels.npy"

def plot_2d(points, labels, title, save_path):
    plt.figure(figsize=(8,6))

    for label, name in [(0, "OK"), (1, "Defective")]:
        mask = labels == label

        plt.scatter(
            points[mask, 0], points[mask, 1],
            label=name, alpha=0.7
        )

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def main():
    embeddings = np.load(EMBEDDINGS_PATH)
    labels = np.load(LABELS_PATH)

    print("Embeddings:", embeddings.shape)
    print("Labels:", labels.shape)

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    print("Running PCA...")

    pca = PCA(n_components=2)
    pca_points = pca.fit_transform(embeddings_scaled)

    print("PCA explained variance ratio:", pca.explained_variance_ratio_)

    plot_2d(
        pca_points, labels, 
        "PCA Visualization of CNN Embeddings",
        "outputs/pca_embeddings.png"
    )

    print("Running t-SNE...")

    tsne = TSNE(
        n_components=2, perplexity=30, init="pca",
        learning_rate="auto", random_state=42
    )

    tsne_points = tsne.fit_transform(embeddings_scaled)

    plot_2d(
        tsne_points, labels, 
        "t-SNE Visualization of CNN Embeddings",
        "outputs/tsne_embeddings.png"
    )

    print("Saved visualizations to outputs/")

if __name__ == "__main__":
    main()