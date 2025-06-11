import numpy as np

def logistic_loss(theta, X, y):
    """
    Numerically stable logistic loss computation
    
    Args:
        theta: model parameters
        X: feature matrix
        y: binary labels in {-1, +1} format
        
    Returns:
        loss: scalar loss value
    """
    # Compute logits
    logits = X @ theta
    
    # For binary classification with labels in {-1, +1}:
    # loss = log(1 + exp(-y * logits))
    z = y * logits
    
    # Use stable computation
    loss = stable_log1pexp(-z)
    
    return np.mean(loss)


def logistic_gradient(theta, X, y):
    """
    Alternative stable logistic gradient computation
    """
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
