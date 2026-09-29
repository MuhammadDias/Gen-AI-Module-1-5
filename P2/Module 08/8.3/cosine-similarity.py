import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two one-dimensional vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def pairwise_similarity(matrix: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity for all rows in a matrix."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = matrix / norms
    return (normalized @ normalized.T).astype(np.float32)


rng = np.random.default_rng(42)
vectors = rng.standard_normal((4, 8)).astype(np.float32)
similarity_matrix = pairwise_similarity(vectors)
print("Pairwise similarities:")
for first in range(4):
    for second in range(first + 1, 4):
        print(
            f"  vec[{first}] vs vec[{second}]: "
            f"{similarity_matrix[first, second]:.4f}"
        )
print(f"Diagonal: {np.diag(similarity_matrix)}")
