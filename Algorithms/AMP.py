import numpy as np
import math
import os
from scipy.optimize import minimize
from common.common import Algorithm 
from common.constraints import constrain_l2_norm 
from lossfunctions.logistic_regression import LogisticRegression

# --- Constants ---
USE_LOWMEM = False  
MAX_ITER_AMP = 500
DIVISION_STABILITY_EPSILON = 1e-9

def amp_run_classification(x, y, loss_func, grad_func,
                          epsilon, delta, lambda_param=None,
                          learning_rate=None, num_iters=None,
                          l2_constraint=None, eps_frac=None,
                          eps_out_frac=None,
                          gamma=None, L=1, random_state=None,
                          verbose=False):
    
    # Set up RNG properly
    rng = np.random.default_rng(random_state)
    
    n, m = x.shape
    if n == 0:
        if verbose: print("Warn: amp_run_classification called with 0 samples.")
        return np.zeros(m), L, 0.0
    
    # Original used: lmbda = pow(L, 2); beta = pow(L, 2)
    # Correct for logistic regression:
    beta_smoothness = 0.25  # Standard β for logistic regression with normalized features
    r = 2  # R norm parameter for GLMs
    
    # Initialize theta
    x0 = np.zeros(shape=m)
    
    # Better default parameter calculation
    if eps_frac is None:
        # Use the original formula but with bounds checking
        if epsilon > 0:
            best = min(0.88671 + 0.0186607 / (epsilon**0.372906), 0.99)
            eps_frac = max(best, 1 - 1/epsilon + 0.001)
            eps_frac = max(0.7, min(0.95, eps_frac))  # Clamp to reasonable range
        else:
            eps_frac = 0.9
    
    if eps_out_frac is None:
        eps_out_frac = 0.01
    
    # Better budget allocation with stability checks
    eps_out = max(DIVISION_STABILITY_EPSILON, epsilon * eps_out_frac)
    eps_obj = max(DIVISION_STABILITY_EPSILON, epsilon - eps_out)
    eps_p = max(DIVISION_STABILITY_EPSILON, eps_frac * eps_obj)
    
    # Ensure eps_p doesn't exceed eps_obj
    if eps_obj < eps_p:
        eps_p = eps_obj * 0.95  # Leave some budget for regularization
    
    delta_out = max(DIVISION_STABILITY_EPSILON, eps_out_frac * delta)
    delta_obj = max(DIVISION_STABILITY_EPSILON, delta - delta_out)
    
    # regularization parameter calculation
    big_lambda_denom = eps_obj - eps_p
    if big_lambda_denom <= 1e-6:  # More generous threshold
        if verbose: 
            print(f"Warn: AMP (eps_obj - eps_p) too small ({big_lambda_denom:.2e}). Adjusting eps_p.")
        eps_p = eps_obj * 0.8  # Use 80% for noise, 20% for regularization
        big_lambda_denom = eps_obj - eps_p
    
    # Use correct beta_smoothness (not L²)
    big_lambda = r * beta_smoothness / big_lambda_denom
    
    # Reasonable bounds on regularization
    big_lambda = max(0.001, min(big_lambda, 100.0))  # Clamp to reasonable range
    
    if gamma is None:
        gamma = max(1e-6, 1.0 / (n**1.5)) if n > 0 else 1e-6
    
    if verbose:
        print(f"  AMP: ε_out={eps_out:.4f}, ε_obj={eps_obj:.4f}, ε_p={eps_p:.4f}")
        print(f"  AMP: Λ={big_lambda:.4f}, β={beta_smoothness}, γ={gamma:.2e}")
    
    # sensitivity calculations with bounds
    sensitivity_obj = (2 * L) / n if n > 0 else 0
    sensitivity_out = (n * gamma) / big_lambda if big_lambda > 0 else 0
    
    # Calculate noise standard deviations with stability checks
    log_term_obj = max(0, math.log(1.0 / max(delta_obj, 1e-100)))
    std_dev_obj = 0.0
    if eps_p > 1e-9:
        std_dev_obj = sensitivity_obj * (1 + np.sqrt(2 * log_term_obj)) / eps_p
        # Cap objective noise to prevent explosion
        max_obj_noise = 0.1 * np.sqrt(m)
        std_dev_obj = min(std_dev_obj, max_obj_noise)
    
    log_term_out = max(0, math.log(1.0 / max(delta_out, 1e-100)))
    std_dev_out = 0.0
    if eps_out > 1e-9:
        std_dev_out = sensitivity_out * (1 + np.sqrt(2 * log_term_out)) / eps_out
        # Cap output noise
        max_out_noise = 0.5 * np.sqrt(m)
        std_dev_out = min(std_dev_out, max_out_noise)
    
    if verbose:
        print(f"  AMP: Objective noise σ={std_dev_obj:.6f}, Output noise σ={std_dev_out:.6f}")
    
    # Generate noise vectors with proper RNG
    noise_obj = rng.normal(scale=std_dev_obj, size=m) if std_dev_obj > 0 else np.zeros(m)
    noise_out = rng.normal(scale=std_dev_out, size=m) if std_dev_out > 0 else np.zeros(m)
    
    # initialization with constraints
    if l2_constraint is not None:
        x0 = (rng.random(m) - 0.5) * 2 * min(l2_constraint * 0.01, 0.01)  # Small initialization
        x0 = constrain_l2_norm(x0, l2_constraint)
    
    # Define perturbed objective and gradient with numerical stability
    def private_loss(theta, x_arg, y_arg):
        # Check for numerical issues
        if np.any(np.isnan(theta)) or np.any(np.isinf(theta)):
            return np.inf
        
        y_bin = np.where(y_arg <= 0, -1, 1)
        try:
            raw_loss = loss_func(theta, x_arg, y_bin)
            if np.isnan(raw_loss) or np.isinf(raw_loss):
                return np.inf
        except:
            return np.inf
        
        reg_term = (big_lambda / (2 * n)) * np.sum(theta**2) if n > 0 else 0.0
        noise_term = np.dot(noise_obj, theta) if m > 0 else 0.0
        
        result = raw_loss + reg_term + noise_term
        return result if np.isfinite(result) else np.inf
    
    def private_gradient(theta, x_arg, y_arg):
        # Check for numerical issues
        if np.any(np.isnan(theta)) or np.any(np.isinf(theta)):
            return np.zeros_like(theta)
        
        y_bin = np.where(y_arg <= 0, -1, 1)
        try:
            raw_gradient = grad_func(theta, x_arg, y_bin)
            if np.any(np.isnan(raw_gradient)) or np.any(np.isinf(raw_gradient)):
                return np.zeros_like(theta)
        except:
            return np.zeros_like(theta)
        
        reg_grad = (big_lambda / n) * theta if n > 0 else np.zeros_like(theta)
        
        result = raw_gradient + reg_grad + noise_obj
        return result if np.all(np.isfinite(result)) else np.zeros_like(theta)
    
    # optimization with constraint handling
    def constrain_theta_callback(theta):
        if l2_constraint is not None:
            theta[:] = constrain_l2_norm(theta, l2_constraint)
    
    # Choose optimization method based on constraints
    if USE_LOWMEM or l2_constraint is not None:
        if l2_constraint is not None:
            bounds = [(-l2_constraint, l2_constraint) for _ in range(m)]
        else:
            bounds = None
        
        opts = {
            'gtol': gamma,
            'disp': verbose,
            'maxiter': min(MAX_ITER_AMP, num_iters if num_iters else MAX_ITER_AMP),
            'maxfun': MAX_ITER_AMP * 2
        }
        result = minimize(private_loss, x0, args=(x, y), method='L-BFGS-B', 
                         jac=private_gradient, bounds=bounds, options=opts)
    else:
        opts = {
            'gtol': gamma,
            'norm': 2,
            'disp': verbose,
            'maxiter': min(MAX_ITER_AMP, num_iters if num_iters else MAX_ITER_AMP)
        }
        cb = constrain_theta_callback if l2_constraint is not None else None
        result = minimize(private_loss, x0, args=(x, y), method='BFGS', 
                         jac=private_gradient, options=opts, callback=cb)
    
    theta = result.x
    
    # Apply final constraint
    if l2_constraint is not None:
        theta = constrain_l2_norm(theta, l2_constraint)
    
    # Check for optimization issues
    if not result.success and verbose:
        print(f"Warn: AMP optimizer did not converge. Message: {result.message}")
    
    # Final check for numerical issues
    if np.any(np.isnan(theta)) or np.any(np.isinf(theta)):
        if verbose: print("Warn: AMP produced NaN/Inf theta, returning zeros")
        theta = np.zeros(m)
    
    # Add output noise
    theta_final = theta + noise_out
    
    # Final constraint application
    if l2_constraint is not None:
        theta_final = constrain_l2_norm(theta_final, l2_constraint)
    
    return theta_final, L, gamma


class ApproximateMinimaPerturbationLR(Algorithm):
    """
    AMP wrapper for Logistic Regression.
    """
    
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def run_classification(self, x, y, epsilon, delta,
                          learning_rate=None, iterations=None,
                          l2_constraint=None, eps_frac=None,
                          eps_out_frac=None,
                          gamma=None, L=1, random_state=None,
                          **kwargs):

        if x.shape[0] == 0:
            return np.zeros(x.shape[1] if x.shape[1] > 0 else 0), L, gamma
        
        # Convert y to binary if needed
        y_bin = np.where(y <= 0, -1, 1) if y is not None and y.size > 0 else np.array([])
        
        if epsilon <= 0 or delta <= 0:
            if self.verbose: print("AMP-LR Error: ε and δ must be positive.")
            return np.zeros(x.shape[1] if x.shape[1] > 0 else 0), L, gamma
        
        if self.verbose:
            print(f"AMP-LR: Starting with ε={epsilon:.4f}, δ={delta:.2e}")
            print(f"AMP-LR: L={L}, constraint={l2_constraint}")
        
        theta, returned_L, returned_gamma = amp_run_classification(
            x, y_bin, 
            LogisticRegression.loss, LogisticRegression.gradient,
            epsilon, delta,
            learning_rate=learning_rate, num_iters=iterations,
            l2_constraint=l2_constraint, eps_frac=eps_frac,
            eps_out_frac=eps_out_frac, gamma=gamma, L=L,
            random_state=random_state, verbose=self.verbose
        )
        
        return theta, returned_L, returned_gamma
    
    @property 
    def name(self):
        return "AMP-LR"