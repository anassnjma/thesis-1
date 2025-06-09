import numpy as np

def stable_logsumexp(x):
    """Numerically stable log-sum-exp"""
    x_max = np.max(x, axis=-1, keepdims=True)
    return x_max + np.log(np.sum(np.exp(x - x_max), axis=-1, keepdims=True))

def stable_log1pexp(x):
    """Numerically stable log(1 + exp(x))"""
    # Use different formulations based on the value of x to avoid overflow
    return np.where(x > 0, 
                    x + np.log1p(np.exp(-x)),  # For x > 0: x + log(1 + exp(-x))
                    np.log1p(np.exp(x)))       # For x <= 0: log(1 + exp(x))

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

def logistic_gradient_stable(theta, X, y):
    """
    Numerically stable logistic gradient computation
    
    Args:
        theta: model parameters  
        X: feature matrix
        y: binary labels in {-1, +1} format
        
    Returns:
        gradient: gradient vector
    """
    # Compute logits
    logits = X @ theta
    z = y * logits
    
    # Stable sigmoid computation: 1 / (1 + exp(-z))
    # Using the identity: sigmoid(z) = exp(z) / (1 + exp(z)) = 1 / (1 + exp(-z))
    sigmoid_z = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))  # Clip to prevent overflow
    
    # Gradient: -X^T * y * (1 - sigmoid(y * logits))
    gradient = -X.T @ (y * (1 - sigmoid_z)) / X.shape[0]
    
    return gradient

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
