import numpy as np

def t_test_one_sample(x, mu0):
    x = np.array(x, dtype=float)
    
    n = x.size
    if n < 2:
        return None
    
    # Mean
    x_mean = np.mean(x)
    
    # Sample std (Bessel correction)
    s = np.std(x, ddof=1)
    
    # Handle zero variance
    if s == 0:
        return 0.0 if x_mean == mu0 else np.inf * np.sign(x_mean - mu0)
    
    # t-statistic
    t_stat = (x_mean - mu0) / (s / np.sqrt(n))
    
    return float(t_stat)