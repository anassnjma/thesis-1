import numpy as np

def logistic_loss(theta, X, y):
    logits = X @ theta
    z = y * logits
    loss = np.log1p(np.exp(-z))  
    return np.mean(loss)


def logistic_gradient(theta, X, y):
    logits = X @ theta
    
    # Clip logits to prevent overflow
    logits_clipped = np.clip(logits, -500, 500)
    
    # Compute sigmoid in a stable way
    sigmoid = np.where(logits_clipped >= 0,
                       1 / (1 + np.exp(-logits_clipped)),
                       np.exp(logits_clipped) / (1 + np.exp(logits_clipped)))
    
    # Convert y from {-1, +1} to {0, 1} for standard gradient computation
    y_01 = (y + 1) / 2
    
    # Standard logistic regression gradient: X^T * (sigmoid - y)
    gradient = X.T @ (sigmoid - y_01) / X.shape[0]
    
    return gradient
