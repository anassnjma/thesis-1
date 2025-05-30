import numpy as np
import math
from scipy.sparse import issparse # Re-added for sparse matrix handling

# These names are expected to be imported from your project's common modules.
from common.common import Algorithm
from common.constraints import constrain_l2_norm

# --- Constants ---
DIVISION_STABILITY_EPSILON = 1e-9
LOG_TERM_CONSTANT_WU = 1.25

def logistic_loss_gradient_regularized_batch(theta, batch_x, batch_y, lambda_param):
    """Computes regularized logistic loss gradient for a batch."""
    num_in_batch = batch_x.shape[0]
    if num_in_batch == 0:
        return np.zeros_like(theta)

    z = batch_x.dot(theta)
    y_times_z_clipped = np.clip(batch_y * z, -500, 500)
    # The minus sign in 1/(1+exp(-y_times_z_clipped)) is standard for logistic loss function formula.
    p = 1 / (1 + np.exp(-y_times_z_clipped))

    if issparse(batch_x): # Check if batch_x is a sparse matrix
        # Gradient term -y(1-p) is correct with the above, because the derivative of the logistic loss is negative
        coefficients = -(batch_y * (1 - p))
        avg_data_gradient = (batch_x.T.multiply(coefficients)).sum(axis=1) / num_in_batch
        avg_data_gradient = np.asarray(avg_data_gradient).squeeze()
    else:
        # Gradient term -y(1-p) is correct.
        data_gradient_terms = -(batch_y * (1 - p))[:, np.newaxis] * batch_x
        avg_data_gradient = data_gradient_terms.sum(axis=0) / num_in_batch

    return avg_data_gradient + lambda_param * theta

def permutation_sgd_wu_etal(X_train, y_train, lambda_param, num_epochs, batch_size_param, rng):
    """
    SGD with data permutation, specific learning rate, and L2 projection. Approximates Wu et al. (2016).
    Relies on imported 'constrain_l2_norm'.
    """
    n_samples, n_features = X_train.shape
    if n_samples == 0: return np.zeros(n_features)
    if lambda_param <= 0: raise ValueError("lambda_param must be > 0.")

    theta = np.zeros(n_features)
    projection_radius = 1.0 / lambda_param
    t_iteration_counter = 1

    # num_epochs corresponds to 'k' (number of passes) in Wu et al.
    for epoch in range(num_epochs):
        permutation_indices = rng.permutation(n_samples)
        X_permuted = X_train[permutation_indices]
        y_permuted = y_train[permutation_indices]

        batch_definitions = []
        # Batching strategy: Ensures all batches are >= batch_size_param (if n_samples >= batch_size_param)
        # by adding remainder to the last of (n_samples // batch_size_param) batches.
        # Each sample is processed once per epoch.
        if n_samples > 0:
            if n_samples < batch_size_param:
                batch_definitions.append((0, n_samples))
            else:
                num_batches_to_form = n_samples // batch_size_param
                current_pos = 0
                for i in range(num_batches_to_form):
                    start = current_pos
                    end = n_samples if i == num_batches_to_form - 1 else start + batch_size_param
                    batch_definitions.append((start, end))
                    current_pos = end
        
        for start_idx, end_idx in batch_definitions:
            if start_idx >= end_idx: continue

            current_batch_X = X_permuted[start_idx:end_idx]
            current_batch_y = y_permuted[start_idx:end_idx]

            # Learning rate based on Wu et al. (2016) Alg 2: min(1/beta, 1/(gamma*t)).
            lr_decreasing = 1.0 / (lambda_param * t_iteration_counter + DIVISION_STABILITY_EPSILON)
            lr_constant = 1.0 / (1.0 + lambda_param + DIVISION_STABILITY_EPSILON)
            learning_rate = min(lr_constant, lr_decreasing)

            gradient = logistic_loss_gradient_regularized_batch(
                theta, current_batch_X, current_batch_y, lambda_param)
            theta = theta - learning_rate * gradient
            theta = constrain_l2_norm(theta, projection_radius) # Uses imported function
            t_iteration_counter += 1
    return theta

class DPSGDLR(Algorithm): # Inherits from imported Algorithm
    """
    DPSGD for Logistic Regression via Output Perturbation (Wu et al. 2016).
    """
    # Assuming Algorithm base class expects instance methods
    def run_classification(self, x, y, epsilon, delta, lambda_param, num_epochs, batch_size, L,
                           random_state=None, **kwargs):
        """
        Args:
            L: Data clipping bound (e.g., L=1 for sensitivity formula).
            random_state: Seed for RNG (for reproducibility).
        """
        rng = np.random.default_rng(random_state)

        n_samples, n_features = x.shape

        if not (lambda_param > 0 and epsilon > 0 and delta > 0 and batch_size > 0):
            print(f"Warn (DPSGDLR): Invalid params. lambda={lambda_param}, eps={epsilon}, delta={delta}, batch_size={batch_size}. Ret 0s.")
            return np.zeros(n_features), L, 0.0
        if n_samples == 0:
             return np.zeros(n_features), L, 0.0

        w_star = permutation_sgd_wu_etal(x, y, lambda_param, num_epochs, batch_size, rng)

        if epsilon == np.inf: return w_star, L, 0.0

        effective_b = min(batch_size, n_samples) if n_samples > 0 else batch_size
        if effective_b == 0:
            print("Warn (DPSGDLR): effective_b for sensitivity is zero. Ret non-private w_star.")
            return w_star, L, 0.0
        
        # Sensitivity assumes ||x_i|| <= 1 (L=1 implicitly).
        sensitivity_w_star = 4.0 / (n_samples * effective_b * lambda_param)

        log_term_val = 0
        if 0 < delta < LOG_TERM_CONSTANT_WU:
            log_term_val = math.log(LOG_TERM_CONSTANT_WU / delta)
        elif delta >= LOG_TERM_CONSTANT_WU:
             print(f"Warn (DPSGDLR): Delta {delta} >= {LOG_TERM_CONSTANT_WU}. Log term non-positive.")
        
        noise_std_dev = 0.0
        if log_term_val > 0 and epsilon > 0:
            noise_std_dev = (sensitivity_w_star * math.sqrt(2.0 * log_term_val)) / epsilon
        
        if noise_std_dev <= 0:
            print(f"Warn (DPSGDLR): Noise_std_dev <=0 (val: {noise_std_dev}). No noise added.")
            noise = np.zeros_like(w_star)
        else:
            noise = rng.normal(loc=0, scale=noise_std_dev, size=w_star.shape)
        
        return w_star + noise, L, 0.0

    @property # Assuming Algorithm base class expects a property
    def name(self):
        return "DPSGD-LR"