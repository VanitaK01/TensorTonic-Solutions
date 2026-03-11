import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def train_logistic_regression(X, y, lr=0.01, steps=1000):
    
    X = np.array(X)
    y = np.array(y)
    
    n_samples, n_features = X.shape
    
    # Initialize parameters
    w = np.zeros(n_features)
    b = 0.0
    
    for _ in range(steps):
        
        # Linear combination
        z = np.dot(X, w) + b
        
        # Sigmoid activation
        y_pred = sigmoid(z)
        
        # Gradients
        dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
        db = (1/n_samples) * np.sum(y_pred - y)
        
        # Update parameters
        w -= lr * dw
        b -= lr * db
    
    return w, b