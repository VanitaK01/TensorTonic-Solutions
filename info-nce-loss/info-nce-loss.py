import numpy as np

def info_nce_loss(Z1, Z2, temperature=0.1):
    # Convert to NumPy arrays
    Z1 = np.array(Z1, dtype=float)
    Z2 = np.array(Z2, dtype=float)
    
    # Validate shapes
    if Z1.ndim != 2 or Z2.ndim != 2 or Z1.shape != Z2.shape:
        return None
    
    if temperature <= 0:
        return None
    
    N = Z1.shape[0]
    
    # Similarity matrix (N x N)
    S = (Z1 @ Z2.T) / temperature
    
    # Numerical stability: subtract row-wise max
    S_max = np.max(S, axis=1, keepdims=True)
    S_stable = S - S_max
    
    # Exponentiate
    exp_S = np.exp(S_stable)
    
    # Denominator: sum over rows
    denom = np.sum(exp_S, axis=1)
    
    # Numerator: diagonal elements (positive pairs)
    numer = np.diag(exp_S)
    
    # Compute loss
    loss = -np.mean(np.log(numer / denom))
    
    return float(loss)