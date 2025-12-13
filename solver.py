import numpy as np

def normalize_ssd(ssd_matrix):
    """
    Normalizes SSD errors to a similarity score [0, 1].
    SSD is a dissimilarity metric (0 is best).
    We invert it so 1 is best.
    Formula: 1 - (x / max(x))
    """
    ssd_matrix = np.array(ssd_matrix)
    if ssd_matrix.size == 0:
        return ssd_matrix
    
    max_val = np.max(ssd_matrix)
    if max_val == 0:
        # If all errors are 0, then all are perfect matches (score 1)
        return np.ones_like(ssd_matrix, dtype=float)
        
    return 1.0 - (ssd_matrix / max_val)

def normalize_correlation(corr_matrix):
    """
    Normalizes correlation coefficients [-1, 1] to a similarity score [0, 1].
    Correlation is a similarity metric (1 is best).
    Formula: (x + 1) / 2
    """
    corr_matrix = np.array(corr_matrix)
    # Ensure it's within expected bounds just in case
    return (np.clip(corr_matrix, -1.0, 1.0) + 1.0) / 2.0

def normalize_gradient(grad_matrix):
    """
    Normalizes gradient errors to a similarity score [0, 1].
    Gradient error is a dissimilarity metric (0 is best).
    Formula: 1 - (x / max(x))
    """
    grad_matrix = np.array(grad_matrix)
    if grad_matrix.size == 0:
        return grad_matrix
        
    max_val = np.max(grad_matrix)
    if max_val == 0:
        return np.ones_like(grad_matrix, dtype=float)
        
    return 1.0 - (grad_matrix / max_val)

def calculate_composite_score(metrics, weights=None):
    """
    Calculates the weighted sum of normalized metrics.
    
    Args:
        metrics (dict): Dictionary where keys are metric names and values are 
                        normalized score matrices (numpy arrays).
                        e.g., {'ssd': ssd_scores, 'grad': grad_scores}
        weights (dict): Dictionary of weights for each metric. 
                        e.g., {'ssd': 1.0, 'grad': 2.0}
                        If None, equal weights are used.
                        
    Returns:
        numpy.ndarray: The combined score matrix.
    """
    if not metrics:
        return None
        
    # Get shape from the first metric
    first_key = list(metrics.keys())[0]
    shape = metrics[first_key].shape
    
    total_score = np.zeros(shape)
    total_weight = 0.0
    
    for name, matrix in metrics.items():
        weight = 1.0
        if weights and name in weights:
            weight = weights[name]
            
        total_score += matrix * weight
        total_weight += weight
        
    if total_weight == 0:
        return np.zeros(shape)
        
    return total_score / total_weight
