import numpy as np
import math
import os
from scipy.optimize import minimize
from common.common import Algorithm 
from common.constraints import constrain_l2_norm 
from lossfunctions.logistic_regression import LogisticRegression 

# --- Module Configuration & Constants ---
USE_LOWMEM_OPTIMIZER = False
OPTIMIZER_MAX_ITERATIONS = 500
OPTIMIZER_GRAD_TOLERANCE = 1e-5
DIVISION_STABILITY_EPSILON = 1e-9
LOG_STABILITY_EPSILON = 1e-9
AMP_R_NORM = 2 # L2 norm

def amp_run_classification(
    x_features, y_labels, loss_function, gradient_function,
    epsilon_total, delta_total, num_iters=None,
    l2_norm_constraint=None, eps_frac_obj_noise_calc=0.9, eps_frac_output_noise=0.01,
    gamma_stability=None, L_grad_bound=1.0):
    """
    Implements Approximate Minima Perturbation (AMP).
    Perturbs objective, minimizes, then perturbs output weights.

    Args:
        x_features, y_labels: Training data (y_labels in {-1, 1}).
        loss_function, gradient_function: SUMMED loss/gradient functions.
        epsilon_total, delta_total: Total privacy budget.
        num_iters: Max iterations for optimizer.
        l2_norm_constraint: Hard L2 norm bound on weights.
        eps_frac_obj_noise_calc: Fraction of (eps_obj) for eps_p (objective noise).
        eps_frac_output_noise: Fraction of epsilon_total for output noise.
        gamma_stability: AMP stability parameter (default 1/n_samples).
        L_grad_bound: Bound on per-sample gradient L2 norm.

    Returns: (private_theta, L_val, gamma_val)
    """
    n_samples, n_features = x_features.shape
    if n_samples == 0: return np.zeros(n_features), L_grad_bound, 0.0

    # --- 1. Parameter Init & Privacy Budget Allocation ---
    beta_param = L_grad_bound**2
    theta_initial = np.zeros(n_features)
    if l2_norm_constraint:
        theta_initial = (np.random.rand(n_features) - 0.5) * 2 * l2_norm_constraint
        theta_initial = constrain_l2_norm(theta_initial, l2_norm_constraint)

    eps_out = epsilon_total * eps_frac_output_noise
    eps_obj_stage = max(0, epsilon_total - eps_out)
    eps_p_privacy = max(0, min(eps_frac_obj_noise_calc * eps_obj_stage, eps_obj_stage))

    delta_out = delta_total * eps_frac_output_noise
    delta_obj_stage = max(0, delta_total - delta_out)

    # --- 2. AMP Internal Regularizer (big_lambda) & Gamma ---
    big_lambda_denom = eps_obj_stage - eps_p_privacy
    if big_lambda_denom <= DIVISION_STABILITY_EPSILON:
        print(f"Warn (AMP): big_lambda_denom near zero ({big_lambda_denom:.1e}). big_lambda may be large.")
        big_lambda_reg = AMP_R_NORM * beta_param / DIVISION_STABILITY_EPSILON
    else:
        big_lambda_reg = (AMP_R_NORM * beta_param) / big_lambda_denom
    
    current_gamma = gamma_stability if gamma_stability is not None else (1.0/n_samples if n_samples > 0 else 1.0)

    # --- 3. Sensitivities & Noise StdDevs ---
    sens_obj_noise = (2 * L_grad_bound) / n_samples if n_samples > 0 else 0.0
    sens_out_stage = (n_samples * current_gamma) / max(big_lambda_reg, DIVISION_STABILITY_EPSILON)

    std_dev_obj, std_dev_out = 0.0, 0.0
    if eps_p_privacy > DIVISION_STABILITY_EPSILON:
        log_term_obj = math.log(1.0 / max(delta_obj_stage, LOG_STABILITY_EPSILON))
        std_dev_obj = sens_obj_noise * (1 + np.sqrt(max(0, 2 * log_term_obj))) / eps_p_privacy
    if eps_out > DIVISION_STABILITY_EPSILON:
        log_term_out = math.log(1.0 / max(delta_out, LOG_STABILITY_EPSILON))
        std_dev_out = sens_out_stage * (1 + np.sqrt(max(0, 2 * log_term_out))) / eps_out

    np.random.seed(ord(os.urandom(1)))
    noise_obj_vec = np.random.normal(scale=std_dev_obj, size=n_features)
    noise_out_vec = np.random.normal(scale=std_dev_out, size=n_features)

    # --- 4. Define & Minimize Perturbed Objective ---
    def priv_obj(th, x, y):
        return loss_function(th, x, y) + \
               (big_lambda_reg / (2*n_samples)) * np.linalg.norm(th)**2 + \
               noise_obj_vec.dot(th)
    def priv_grad(th, x, y):
        return gradient_function(th, x, y) + \
               (big_lambda_reg / n_samples) * th + \
               noise_obj_vec

    cb = (lambda th_iter: setattr(th_iter, '[::]', constrain_l2_norm(th_iter, l2_norm_constraint))) if l2_norm_constraint else None
    iters = num_iters if num_iters is not None else OPTIMIZER_MAX_ITERATIONS
    opt_method = 'L-BFGS-B' if USE_LOWMEM_OPTIMIZER else 'BFGS'
    opt_opts = {'gtol': OPTIMIZER_GRAD_TOLERANCE, 'disp': False, 'maxiter': iters}
    if opt_method == 'BFGS': opt_opts['norm'] = np.inf

    res = minimize(priv_obj, theta_initial, args=(x_features, y_labels), method=opt_method,
                   jac=priv_grad, options=opt_opts, callback=cb if opt_method == 'BFGS' else None)
    
    theta_intermed = res.x
    if not res.success: print(f"Warn (AMP): Optimizer ({opt_method}) failed: {res.message}")
    if l2_norm_constraint: theta_intermed = constrain_l2_norm(theta_intermed, l2_norm_constraint)

    # --- 5. Add Output Noise ---
    theta_final = theta_intermed + noise_out_vec
    if l2_norm_constraint: theta_final = constrain_l2_norm(theta_final, l2_norm_constraint)
    return theta_final, L_grad_bound, current_gamma

class ApproximateMinimaPerturbationLR(Algorithm):
    """Wrapper for AMP tailored for Logistic Regression."""
    @staticmethod
    def run_classification(x, y, epsilon, delta, iterations=None,
                           l2_constraint=None, eps_frac=0.9, eps_out_frac=0.01,
                           gamma=None, L=1.0): # Removed lambda_param, learning_rate, **kwargs
        """
        Runs AMP for Logistic Regression. See amp_run_classification for arg details.
        Removed unused: lambda_param, learning_rate.
        """
        y_bin = np.where(y <= 0, -1, 1)
        loss_f, grad_f = LogisticRegression.loss, LogisticRegression.gradient

        return amp_run_classification(
            x_features=x, y_labels=y_bin,
            loss_function=loss_f, gradient_function=grad_f,
            epsilon_total=epsilon, delta_total=delta,
            num_iters=iterations,
            l2_norm_constraint=l2_constraint,
            eps_frac_obj_noise_calc=eps_frac,
            eps_frac_output_noise=eps_out_frac,
            gamma_stability=gamma,
            L_grad_bound=L
        )

    @staticmethod
    def name(): return "AMP-LR"