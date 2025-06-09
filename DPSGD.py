import numpy as np
from lossfunction import logistic_loss, logistic_gradient


def DPSGD(D, epsilon, delta, L, T, k, learning_rate=None):
    """
    Algorithm 2: Differentially Private Minibatch Stochastic Gradient Descent
    
    Args:
        D: tuple (X, y) - feature matrix and labels
        epsilon, delta: privacy parameters
        L: L2-Lipschitz constant of loss function (gradient clipping bound)
        T: number of iterations
        k: minibatch size
        learning_rate: learning rate function η(t) or constant (default: 1/√t)
    
    Returns:
        θ_T: private model parameters
    
    Key hyperparameters: L, T, k, learning_rate
    """
    X, y = D
    n, p = X.shape
    
    # Step 1: σ² ← (16L²T log(1/δ))/(n²ε²)
    sigma_squared = (16 * L**2 * T * np.log(1/delta)) / (n**2 * epsilon**2)
    sigma = np.sqrt(sigma_squared)
    
    # Step 2: θ₁ = 0^p
    theta = np.zeros(p)
    
    # Default learning rate: 1/√t
    if learning_rate is None:
        learning_rate = lambda t: 1.0 / np.sqrt(t + 1)
    elif isinstance(learning_rate, (int, float)):
        # Convert constant to function
        lr_const = learning_rate
        learning_rate = lambda t: lr_const
    
    # Step 3: for t = 1 to T do
    for t in range(T):
        # Step 4: s₁, ..., sₖ ← Sample k samples uniformly with replacement from D
        indices = np.random.choice(n, size=k, replace=True)
        X_batch = X[indices]
        y_batch = y[indices]
        
        # Step 5: Compute gradient ∇ℓ(θ; batch) = (1/k)∑∇ℓ(θ;sᵢ)
        grad_batch = logistic_gradient(theta, X_batch, y_batch)
        
        # Step 6: Clip gradient: ĝ ← grad_batch / max(1, ||grad_batch||₂/L)
        grad_norm = np.linalg.norm(grad_batch)
        if grad_norm > L:
            grad_batch = grad_batch * (L / grad_norm)
        
        # Step 7: b_t ~ N(0, σ²I_{p×p})
        b_t = np.random.normal(0, sigma, p)
        
        # Step 8: θ_{t+1} = θ_t - η(t)[ĝ + b_t]
        eta_t = learning_rate(t)
        theta = theta - eta_t * (grad_batch + b_t)
    
    # Step 9: Output θ_T
    return theta

# Example:
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    n, p = 1000, 5
    X = np.random.randn(n, p)
    true_theta = np.random.randn(p)
    y = np.sign(X @ true_theta + 0.1 * np.random.randn(n))
    
    # Split data into train/test
    train_size = int(0.8 * n)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Test different epsilon values
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    delta = 1e-5
    L = 1.0      # Lipschitz constant
    T = 100      # Number of iterations
    k = 32       # Minibatch size
    
    print("Epsilon\tAccuracy")
    print("-" * 20)
    
    for epsilon in epsilons:
        # Train private model
        theta_private = DPSGD((X_train, y_train), epsilon, delta, L, T, k)
        
        # Predict on test set
        y_pred = np.sign(X_test @ theta_private)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred == y_test)
        print(f"{epsilon:.1f}\t{accuracy:.4f}")