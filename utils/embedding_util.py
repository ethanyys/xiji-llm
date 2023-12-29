import numpy as np


def normalize_embeddings(x):
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x /= x_norm
    return x
