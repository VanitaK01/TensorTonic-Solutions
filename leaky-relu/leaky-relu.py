import numpy as np

def leaky_relu(x, alpha=0.01):
    
    x = np.asarray(x)   # convert list to numpy array
    
    return np.where(x >= 0, x, alpha * x)