import numpy as np
import math
import os
from scipy.sparse import csr_matrix, hstack
from common.common import Algorithm
from common.constraints import constrain_l2_norm
from lossfunctions.logistic_regression import LogisticRegression

class GradientPerturbationDPSGD(Algorithm):
    """
    Gradient Perturbation DP-SGD (Bassily et al. 2014 / Abadi et al. 2016 style).
    
    This implements true gradient perturbation where Gaussian noise is added
    to gradients at each iteration, unlike output perturbation methods.
    
    Based on Algorithm 2 from Iyengar et al. (2019) paper and GitHub implementation.
    """
    
    def run_classification(self, x, y, epsilon, delta, 
                          num_iters=100, learning_rate=0.01, 
                          L=1.0, minibatch_size=50,
                          l2_constraint=None, lambda_param=0,
                          random_state=None, **kwargs):
        """
        Train logistic regression using gradient perturbation DP-SGD.
        
        Args:
            x: Feature matrix (n_samples × n_features)
            y: Labels in {-1, +1} format
            epsilon: Privacy parameter (smaller = more private)
            delta: Privacy parameter (failure probability)
            num_iters: Number of SGD iterations
            learning_rate: Learning rate for SGD
            L: Lipschitz constant / gradient clipping bound
            minibatch_size: Size of minibatches
            l2_constraint: L2 constraint on parameters (None = unconstrained)
            lambda_param: L2 regularization parameter
            random_state: Random seed for reproducibility
        
        Returns:
            tuple: (theta, L, num_iters) for compatibility with other algorithms
        """
        
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
        
        n, m = x.shape
        
        if n == 0 or epsilon <= 0 or delta <= 0:
            return np.zeros(m), L, num_iters
        
        # Determine regularization setup and initialization
        if l2_constraint is None and lambda_param == 0:
            # Unconstrained, unregularized learning
            L_reg = L
            theta = np.zeros(m)
        elif l2_constraint is not None and lambda_param > 0:
            # Constrained, regularized learning
            L_reg = L + lambda_param * l2_constraint
            theta = (np.random.rand(m) - 0.5) * 2 * l2_constraint
        else:
            # Invalid combination - return zeros
            return np.zeros(m), L, num_iters
        
        # Calculate noise standard deviation according to theory
        # σ = 4 * L_reg * √(T * ln(1/δ)) / (n * ε)
        std_dev = 4 * L_reg * math.sqrt(num_iters * math.log(1/max(delta, 1e-100))) / (n * epsilon)
        
        # Prepare data for minibatch sampling
        if isinstance(x, csr_matrix):
            # Handle sparse matrices
            data = csr_matrix(hstack((x, csr_matrix(y.reshape(-1, 1)))))
        else:
            # Handle dense matrices
            data = np.column_stack((x, y))
        
        # Main training loop with gradient perturbation
        for i in range(num_iters):
            # Sample minibatch with replacement
            minibatch_indices = np.random.choice(data.shape[0], minibatch_size, replace=True)
            
            if isinstance(x, csr_matrix):
                # Handle sparse data
                minibatch = data[minibatch_indices]
                minibatch_x = minibatch[:, :-1]
                minibatch_y = minibatch[:, -1]
                minibatch_y = np.squeeze(np.asarray(minibatch_y.todense()))
            else:
                # Handle dense data
                minibatch = data[minibatch_indices]
                minibatch_x = minibatch[:, :-1]
                minibatch_y = minibatch[:, -1]
            
            # Compute gradient on minibatch
            gradient = self._compute_logistic_gradient(
                theta, minibatch_x, minibatch_y, lambda_param
            )
            
            # Add Gaussian noise to gradient
            noise = np.random.normal(scale=std_dev, size=m)
            noisy_gradient = gradient + noise
            
            # Update parameters with noisy gradient
            theta = theta - learning_rate * noisy_gradient
            
            # Project back to constraint set if needed
            if l2_constraint is not None:
                theta = constrain_l2_norm(theta, l2_constraint)
        
        return theta, L, num_iters
    
    def _compute_logistic_gradient(self, theta, batch_x, batch_y, lambda_param):
        """
        Compute logistic regression gradient for a batch.
        Uses the same gradient computation as your existing LogisticRegression class.
        """
        if lambda_param > 0:
            # Use regularized gradient
            return self._regularized_logistic_gradient(theta, batch_x, batch_y, lambda_param)
        else:
            # Use unregularized gradient
            return self._unregularized_logistic_gradient(theta, batch_x, batch_y)
    
    def _unregularized_logistic_gradient(self, theta, x, y):
        """
        Gradient for unregularized logistic regression.
        Based on your LogisticRegression.gradient implementation.
        """
        if x.shape[0] == 0:
            return np.zeros_like(theta)
        
        exponent = y * (x.dot(theta))
        # Clip for numerical stability
        exponent = np.clip(exponent, -500, 500)
        
        gradient_loss = -(x.T @ (y / (1 + np.exp(exponent)))) / x.shape[0]
        
        # Ensure proper shape
        if hasattr(gradient_loss, 'reshape'):
            gradient_loss = gradient_loss.reshape(theta.shape)
        
        return gradient_loss
    
    def _regularized_logistic_gradient(self, theta, x, y, lambda_param):
        """
        Gradient for L2-regularized logistic regression.
        """
        data_gradient = self._unregularized_logistic_gradient(theta, x, y)
        regularization_gradient = lambda_param * theta
        return data_gradient + regularization_gradient
    
    @property
    def name(self):
        return "GradientPerturbation-DPSGD"