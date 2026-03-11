import math

def sigmoid(x):
    
    # If input is a list
    if isinstance(x, list):
        return [sigmoid(i) for i in x]
    
    # Stable computation
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1 + exp_x)