import numpy as np

def entropy_node(y):
    y = np.asarray(y)
    
    # Handle empty input
    if y.size == 0:
        return 0.0
    
    # Class counts
    _, counts = np.unique(y, return_counts=True)
    
    # Probabilities
    probs = counts / counts.sum()
    
    # Remove zeros (numerical stability)
    probs = probs[probs > 0]
    
    # Entropy (base-2)
    entropy = -np.sum(probs * np.log2(probs))
    
    return float(entropy)