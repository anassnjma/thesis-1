import numpy as np
from lossfunction import logistic_gradient
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def DPPSGD(S, k, epsilon, eta, batch_size=1, reg_lambda=0.0):
    """
    Algorithm 1: Private Convex Permutation-based SGD (Wu et al. 2017)
    Extended with mini-batching, L2 regularization, and epsilon,delta-differential privacy.
    
    Args:
        S: dataset
        k: number of passes through the data
        epsilon: privacy parameter
        eta: learning rate - constant across all iterations
        batch_size: mini-batch size (default=1 for original algorithm)
        reg_lambda: L2 regularization parameter (default=0.0)
    
    Returns:
        θ_private: private model parameters
        
    Key tuning parameters:
        1. batch_size: mini-batch size
        2. k: number of passes through data
        3. reg_lambda: L2 regularization strength
    """
    X, y = S
    n, d = X.shape
    delta = 1.0 / (n * n)
    
    # Step 1: Initialize model parameters
    w = np.zeros(d)
    
    # Step 2: Train with Permutation-based SGD (k passes, mini-batching)
    for pass_num in range(k):
        # Create random permutation τ for this pass
        tau = np.random.permutation(n)
        
        # Process data in mini-batches
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            batch_indices = tau[batch_start:batch_end]
            actual_batch_size = len(batch_indices)
            
            # Compute average gradient over mini-batch
            batch_gradient = np.zeros(d)
            for i in batch_indices:
                # Compute gradient for single sample
                gradient = logistic_gradient(w, X[i:i+1], y[i:i+1])
                batch_gradient += gradient
            
            # Average gradients over batch
            batch_gradient /= actual_batch_size
            
            # Add L2 regularization term: λ * w
            if reg_lambda > 0:
                batch_gradient += reg_lambda * w
            
            # Update parameters
            w = w - eta * batch_gradient
    
    # Step 3: Calculate L2-sensitivity (Δ₂ ← 2kLη/batch_size)
    L = 1.0  # Lipschitz constant (assuming preprocessed data)
    Delta_2 = 2 * k * L * eta / batch_size
    
    # Step 4: Sample noise vector κ for (ε,δ)-differential privacy
    # Using Gaussian mechanism (Theorem 3): σ ≥ c*Δ₂/ε where c² > 2ln(1.25/δ)
    c_squared = 2 * np.log(1.25 / delta)
    c = np.sqrt(c_squared)
    sigma = c * Delta_2 / epsilon
    kappa = np.random.normal(0, sigma, d)
    
    # Step 5: Return private model (w + κ)
    theta_private = w + kappa
    
    return theta_private

# Example usage:
if __name__ == "__main__":
    # Generate data
    np.random.seed(42)
    n, p = 1000, 5
    X = np.random.randn(n, p)
    true_theta = np.random.randn(p)
    y = (X @ true_theta + 0.1 * np.random.randn(n) > 0).astype(int)
    
    # Split data
    train_size = int(0.8 * n)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Test different epsilon values
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    k = 10           # Number of passes
    eta = 0.1        # Learning rate
    batch_size = 32  # Batch size
    
    print("Epsilon\tAccuracy")
    print("-" * 20)
    
    for epsilon in epsilons:
        # Train private model
        theta_private = DPPSGD((X_train, y_train), k, epsilon, eta, batch_size)
        
        # Predict and calculate accuracy
        y_pred = (X_test @ theta_private > 0).astype(int)
        accuracy = np.mean(y_pred == y_test)
        
        print(f"{epsilon:.1f}\t{accuracy:.4f}")