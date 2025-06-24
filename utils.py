import numpy as np
from scipy.spatial.distance import cosine, euclidean

def yat_similarity(v1, v2):
    """Calculates the Yat similarity: (dot_product^2) / (euclidean_distance^2)."""
    dot_product = np.dot(v1, v2)
    euclidean_dist_sq = np.sum((v1 - v2)**2)
    return (dot_product**2) / (euclidean_dist_sq + 1e-9)

def cosine_similarity(v1, v2):
    """Calculates cosine similarity."""
    return 1 - cosine(v1, v2)

def euclidean_similarity(v1, v2):
    """Calculates euclidean similarity (inverse of distance)."""
    return 1 / (1 + euclidean(v1, v2))

def dot_product_similarity(v1, v2):
    """Calculates dot product similarity."""
    return np.dot(v1, v2)

def set_seeds(seed):
    """Sets random seeds for reproducibility."""
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
