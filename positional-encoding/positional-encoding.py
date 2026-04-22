import numpy as np

def positional_encoding(seq_len, d_model, base=10000):
    # Positions (seq_len, 1)
    pos = np.arange(seq_len).reshape(-1, 1)
    
    # Even indices (0,2,4,...)
    i = np.arange(0, d_model, 2)
    
    # Compute denominator
    denom = np.power(base, (i / d_model))
    
    # Angles
    angles = pos / denom
    
    # Output
    PE = np.zeros((seq_len, d_model), dtype=float)
    
    # Even columns → sin
    PE[:, 0::2] = np.sin(angles)
    
    # Odd columns → cos
    if d_model > 1:
        PE[:, 1::2] = np.cos(angles[:, :PE[:, 1::2].shape[1]])
    
    return PE