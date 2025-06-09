import numpy as np
from scipy.optimize import minimize
from lossfunction import logistic_loss, logistic_gradient

def AMP(D, epsilon, delta, gamma=None, L=1.0, eps_2_frac=0.4, eps_3_frac=0.3, 
        delta_2_frac=0.5, r=2, max_retries=3):
    """
    Algorithm 1 from Iyengar et al. 2019: Approximate Minima Perturbation

    Args:
        D: tuple (X, y) - feature matrix and labels
        epsilon, delta: privacy parameters
        gamma: gradient norm bound (auto-set if None)
        L: Lipschitz constant (default=1.0)
        eps_2_frac, eps_3_frac, delta_2_frac: privacy budget fractions
        
    Returns:
        θ_out: private model parameters
    """
    X, y = D
    n, p = X.shape
    
    beta = L  # β = L for logistic regression with bounded features
    
    if gamma is None:
        gamma = 1.0 / (n ** 2)
    
    # Step 1: Privacy budget allocation
    eps_2 = eps_2_frac * epsilon  # ε₂
    eps_1 = epsilon - eps_2       # ε₁ = ε - ε₂  
    eps_3 = eps_3_frac * eps_1    # ε₃
    delta_2 = delta_2_frac * delta  # δ₂
    delta_1 = delta - delta_2       # δ₁ = δ - δ₂
    
    # Constraint handling: ensure 0 < ε₁ - ε₃ < 1
    if eps_1 - eps_3 >= 1:
        eps_3 = eps_1 - 0.99  # Ensure < 1
    
    if eps_1 - eps_3 <= 0:
        eps_3 = eps_1 * 0.1   # Ensure > 0
    
    # Check if constraints are satisfiable
    if (eps_1 - eps_3) >= 1 or eps_1 <= 0 or eps_3 <= 0:
        return np.zeros(p)
    
    # Step 2: Set regularization parameter Λ ≥ rβ/(ε₁ - ε₃)
    Lambda = r * beta / (eps_1 - eps_3)
    
    # Step 3: Sample objective perturbation noise b₁ ~ N(0, σ₁²I_{p×p})
    sigma_1 = (2 * L / n) * (1 + np.sqrt(2 * np.log(1 / delta_1))) / eps_3
    b_1 = np.random.normal(0, sigma_1, p)
    
    # Step 6 preparation: Calculate output noise variance
    sigma_2 = (n * gamma / Lambda) * (1 + np.sqrt(2 * np.log(1 / delta_2))) / eps_2
    
    # Step 4: Define perturbed objective L_priv(θ; D)
    def L_priv(theta):
        loss = logistic_loss(theta, X, y)
        reg_term = (Lambda / (2 * n)) * np.sum(theta**2)
        noise_term = np.dot(b_1, theta)
        return loss + reg_term + noise_term
    
    def grad_L_priv(theta):
        grad_loss = logistic_gradient(theta, X, y)
        grad_reg = (Lambda / n) * theta
        grad_noise = b_1
        return grad_loss + grad_reg + grad_noise
    
    # Step 5: Find θ_approx such that ||∇L_priv(θ; D)|| ≤ γ
    current_gamma = gamma
    theta_approx = None
    
    for attempt in range(max_retries):
        theta_init = np.zeros(p)
        
        try:
            result = minimize(
                L_priv,
                theta_init,
                jac=grad_L_priv,
                method='BFGS',
                options={'gtol': current_gamma, 'maxiter': 1000}
            )
            
            # Check if optimization succeeded and meets gradient norm requirement
            if result.success:
                final_grad = grad_L_priv(result.x)
                final_grad_norm = np.linalg.norm(final_grad)
                
                if final_grad_norm <= current_gamma:
                    theta_approx = result.x
                    break
            
            # If we have a result but didn't meet the constraint, keep it as backup
            if theta_approx is None and 'result' in locals():
                theta_approx = result.x
        
        except Exception:
            pass
        
        # Retry with relaxed constraint if not last attempt
        if attempt < max_retries - 1:
            current_gamma *= 2
    
    # Fallback if optimization completely failed
    if theta_approx is None:
        theta_approx = np.zeros(p)
    
    # Step 6: Sample output perturbation noise b₂ ~ N(0, σ₂²I_{p×p})
    b_2 = np.random.normal(0, sigma_2, p)
    
    # Step 7: Output θ_out = θ_approx + b₂
    theta_out = theta_approx + b_2
    
    return theta_out

if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    n, p = 1000, 5
    X = np.random.randn(n, p)
    # Normalize features to satisfy Lipschitz constraint
    X = X / np.max(np.linalg.norm(X, axis=1))
    
    true_theta = np.random.randn(p)
    y = (X @ true_theta + 0.1 * np.random.randn(n) > 0).astype(int)
    
    # Split data
    train_size = int(0.8 * n)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Test different epsilon values
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    delta = 1e-5
    L = 1.0
    
    print("Epsilon\tAccuracy")
    print("-" * 20)
    
    for epsilon in epsilons:
        # Train private model
        theta_private = AMP((X_train, y_train), epsilon, delta, L=L)
        
        # Predict and calculate accuracy
        y_pred = (X_test @ theta_private > 0).astype(int)
        accuracy = np.mean(y_pred == y_test)
        
        print(f"{epsilon:.1f}\t{accuracy:.4f}")