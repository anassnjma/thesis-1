import numpy as np
import math
from scipy.sparse import issparse
from common.common import Algorithm 
from common.constraints import constrain_l2_norm 

# --- Constants ---
DIVISION_STABILITY_EPSILON = 1e-9
LOG_TERM_CONSTANT_WU = 1.25

def logistic_loss_gradient_regularized_batch(theta, batch_x, batch_y, lambda_param):
    """Computes regularized logistic loss gradient for a batch."""
    if batch_x.shape[0] == 0: return np.zeros_like(theta)

    z = batch_x.dot(theta)
    y_times_z_clipped = np.clip(batch_y * z, -500, 500) # Stability for exp
    p = 1 / (1 + np.exp(-y_times_z_clipped)) # RR: Do you need a minus sign here?

    if issparse(batch_x):
        coefficients = -(batch_y * (1 - p)) # RR: Check, if you hadn't included a minus sign above,
                                            # could write -batch_y/p
        avg_data_gradient = (batch_x.T.multiply(coefficients)).mean(axis=1)
        avg_data_gradient = np.asarray(avg_data_gradient).squeeze()
    else:
        data_gradient_terms = -(batch_y * (1 - p))[:, np.newaxis] * batch_x # RR: Same here.
        avg_data_gradient = data_gradient_terms.mean(axis=0)

    return avg_data_gradient + lambda_param * theta

def permutation_sgd_wu_etal(X_train, y_train, lambda_param, num_epochs, batch_size=50):
    """
    SGD with data permutation, specific learning rate schedule (1/(lambda_param*t)),
    and L2 projection (radius 1/lambda_param). Approximates Wu et al.
    """
    n_samples, n_features = X_train.shape
    if n_samples == 0: return np.zeros(n_features)
    if lambda_param <= 0: raise ValueError("lambda_param must be > 0.")

    theta = np.zeros(n_features)
    projection_radius = 1.0 / lambda_param
    t_iteration_counter = 1

    for _ in range(num_epochs):
        # RR: So num_epochs is the number of passes k in Wu et al., right?
        permutation_indices = np.random.permutation(n_samples)
        for i_batch in range(math.ceil(n_samples / batch_size)):
            start, end = i_batch * batch_size, min((i_batch + 1) * batch_size, n_samples)
            # RR: Here there is a chance that the last batch has smaller size than batch_size. But this
            # is not allowed if your privacy guarantee is to hold. The batches should each be at least batch_size big.
            # The easy way to achieve this is to merge the last two batches.

            batch_indices = permutation_indices[start:end]
            if not batch_indices.size: continue

            learning_rate = 1.0 / (lambda_param * t_iteration_counter + DIVISION_STABILITY_EPSILON)
            # RR: You need to take the minimum of this and 1/(lambda + 0.25) don't you?

            gradient = logistic_loss_gradient_regularized_batch(theta, X_train[batch_indices], y_train[batch_indices], lambda_param)
            theta = theta - learning_rate * gradient
            theta = constrain_l2_norm(theta, projection_radius)
            t_iteration_counter += 1
    return theta

class DPSGDLR(Algorithm):
    """
    DPSGD for Logistic Regression via Output Perturbation.
    Trains using SGD (approximating Wu et al.), then adds Gaussian noise to weights.
    """
    @staticmethod
    def run_classification(x, y, epsilon, delta, lambda_param, num_epochs, batch_size, L, **kwargs):
        """
        Args:
            x, y: Training data and labels ({-1, 1}).
            epsilon, delta: Privacy parameters.
            lambda_param: L2 regularization for SGD (must be > 0).
            num_epochs, batch_size: SGD parameters.
            L: Data clipping bound (used prior to this call). This L is not directly in sensitivity formula here.
        Returns:
            (private_weights, L_val, 0.0)
        """
        n_samples, n_features = x.shape
        if n_samples == 0: return np.zeros(n_features), L, 0.0
        if not (lambda_param > 0 and epsilon > 0 and delta > 0):
            print(f"Warn (DPSGDLR): Invalid params (lambda_param/eps/delta). lambda={lambda_param}, eps={epsilon}, delta={delta}. Ret 0s.")
            return np.zeros(n_features), L, 0.0

        w_star = permutation_sgd_wu_etal(x, y, lambda_param, num_epochs, batch_size)

        if epsilon == np.inf: return w_star, L, 0.0

        # Sensitivity from Wu et al. context: 4.0 / (n_samples * batch_size * lambda_param)
        if n_samples == 0 or batch_size == 0 or lambda_param == 0: # Should be caught by earlier check for lambda_param
             print("Warn (DPSGDLR): Div by zero in sensitivity. Ret non-private w_star.")
             return w_star, L, 0.0
        sensitivity_w_star = 4.0 / (n_samples * batch_size * lambda_param)

        log_term_val = 0
        if 0 < delta < LOG_TERM_CONSTANT_WU: log_term_val = math.log(LOG_TERM_CONSTANT_WU / delta)
        else: print(f"Warn (DPSGDLR): Delta {delta} unusual. Log term may be 0 or neg.")
        if epsilon <=0: return w_star, L, 0.0 # Should be caught

        noise_std_dev = (sensitivity_w_star * math.sqrt(max(0, 2.0 * log_term_val))) / epsilon
        
        if noise_std_dev <= 0:
            print("Warn (DPSGDLR): Noise std dev is <=0. No noise added.")
            noise = np.zeros_like(w_star)
        else:
            noise = np.random.normal(loc=0, scale=noise_std_dev, size=n_features)
            # RR: It's good practice, for reproducibility, to include a random_state parameter and generate
            # noise seeded by the random_state. E.g., at the start of your script create a random number
            # generator instance rng = np.random.default_rng(seed=12345), make any function that generates random
            # numbers take a rng as argument, and pass in the rng instance you created whenever you call the function.
            # Use the passed in rng to generate random variables, e.g. rng.normal(loc=0, scale=noise_std_dev, size=n_features)
        return w_star + noise, L, 0.0

    @staticmethod
    def name(): return "DPSGD-LR"
