import numpy as np

def normalize(x):
    """normalize a set of vectors to have unit length

    Args:
        x (np.ndarray): array or matrix

    Returns:
        np.ndarray: normalized array or matrix
    """    
    norm = x / np.linalg.norm(x)
    return norm
