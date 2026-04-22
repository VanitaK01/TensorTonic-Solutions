import numpy as np

def cross_entropy_loss(y_true, y_pred):
    # Convert to NumPy arrays (handles list input)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Number of samples
    N = y_true.shape[0]
    
    # Select the predicted probabilities of the correct classes
    probs = y_pred[np.arange(N), y_true]
    
    # Compute loss
    loss = -np.mean(np.log(probs))
    
    return float(loss)