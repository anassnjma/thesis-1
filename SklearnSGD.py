import numpy as np
from sklearn.linear_model import SGDClassifier

def SklearnSGD(D, epsilon, delta, **kwargs):
    """
    Differentially Private Logistic Regression using Output Perturbation.
    
    This function trains a logistic regression model with L2 regularization
    and adds calibrated Gaussian noise to the output coefficients to achieve
    (epsilon, delta)-differential privacy.
    
    Args:
        D: (X, y) tuple of training data
        epsilon: Privacy parameter (epsilon > 0)
        delta: Privacy parameter (0 < delta < 1)
        **kwargs: Hyperparameters including alpha (L2 regularization strength)
        
    Returns:
        private_theta: Differentially private model parameters
    """
    X, y = D
    n, d = X.shape
    
    # Extract regularization parameter - CRITICAL for sensitivity bound
    alpha = kwargs.get('alpha', 0.01)
    
    # Ensure alpha is not too small to avoid numerical issues
    alpha = max(alpha, 1e-6)
    
    # Convert labels from {-1, 1} to {0, 1} for sklearn
    y_sklearn = np.where(y == 1, 1, 0)
    
    # Train logistic regression with L2 regularization
    sgd = SGDClassifier(
        loss='log_loss',
        penalty='l2', 
        alpha=alpha,
        random_state=42,
        max_iter=kwargs.get('max_iter', 1000),  
        tol=1e-3,
        fit_intercept=False,
        learning_rate='constant',
        eta0=kwargs.get('eta0', 0.01)
    )
    
    sgd.fit(X, y_sklearn)
    theta = sgd.coef_.flatten()

    sensitivity = 2.0 / (n * alpha)
    
    # Calculate Gaussian noise scale for (epsilon, delta)-DP
    c = np.sqrt(2 * np.log(1.25 / delta))
    sigma = (c * sensitivity) / epsilon
    
    # Add calibrated Gaussian noise
    noise = np.random.normal(0, sigma, d)
    private_theta = theta + noise
    
    return private_theta
