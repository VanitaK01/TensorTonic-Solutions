import numpy as np

def bernoulli_pmf_and_moments(x, p):
    # Convert input to NumPy array
    x = np.array(x)
    
    # Validate probability
    if not (0 <= p <= 1):
        return None
    
    # Validate x values (must be 0 or 1)
    if not np.all((x == 0) | (x == 1)):
        return None
    
    # Compute PMF
    pmf = np.where(x == 1, p, 1 - p)
    
    # Mean and variance
    mean = float(p)
    var = float(p * (1 - p))
    
    return pmf, mean, var