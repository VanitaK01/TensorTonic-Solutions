import numpy as np

def kl_divergence(p, q, eps=1e-12):
    # Convert to NumPy arrays
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    
    # Validate shapes
    if p.shape != q.shape:
        return None
    
    # Add epsilon to q for numerical stability
    q = q + eps
    
    # Mask where p > 0 (since 0 * log(0/q) = 0)
    mask = p > 0
    
    # Compute KL divergence
    kl = np.sum(p[mask] * np.log(p[mask] / q[mask]))
    
    return float(kl)