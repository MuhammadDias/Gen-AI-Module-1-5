# 5.1 Shape, Reshape, and Indexing
import numpy as np

rng = np.random.default_rng(42)
embeddings = rng.standard_normal((4, 8))
print("Shape:", embeddings.shape)
print("First 3 dims of all docs:\n", embeddings[:, :3])

flat = embeddings.flatten()
back = flat.reshape(4, 8)

similarity_scores = np.array([0.91, 0.43, 0.78, 0.55])
above_threshold = embeddings[similarity_scores > 0.7]
print(f"Docs above 0.7 similarity: {above_threshold.shape[0]}")
