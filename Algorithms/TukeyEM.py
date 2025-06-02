import numpy as np
from common.common import Algorithm

class DPTukey(Algorithm):
    """
    Differentially Private Tukey Depth Regression Algorithm.
    
    This implements the DP-Tukey mechanism for selecting a high-depth model
    from a collection of candidate models using the exponential mechanism
    and propose-test-release (PTR).
    
    Based on the original Tukey depth mechanism with racing sampling.
    """
    
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def run_classification(self, x, y, epsilon, delta, 
                          num_models=20, random_state=None, 
                          **kwargs):
        """
        Train multiple models and select one using DP-Tukey mechanism.
        
        Args:
            x: Feature matrix (n_samples × n_features)
            y: Labels 
            epsilon: Privacy parameter
            delta: Privacy parameter
            num_models: Number of candidate models to train
            random_state: Random seed for reproducibility
            
        Returns:
            tuple: (selected_theta, epsilon, delta) for compatibility
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        n_samples, n_features = x.shape
        
        if n_samples == 0 or epsilon <= 0 or delta <= 0:
            return np.zeros(n_features), epsilon, delta
        
        # Train multiple non-private models on data subsets
        models = self._train_multiple_models(x, y, num_models)
        
        if models.shape[0] < 2:
            if self.verbose:
                print("DPTukey: Insufficient models generated, returning zeros")
            return np.zeros(n_features), epsilon, delta
        
        # Apply DP-Tukey mechanism
        selected_model = self._dp_tukey(models, epsilon, delta)
        
        return selected_model, epsilon, delta
    
    def _train_multiple_models(self, x, y, num_models):
        """Train multiple logistic regression models on random subsets."""
        n_samples, n_features = x.shape
        models = []
        
        # Ensure we have enough data for the requested number of models
        min_samples_per_model = max(10, n_features + 1)
        max_possible_models = n_samples // min_samples_per_model
        actual_num_models = min(num_models, max_possible_models, n_samples // 2)
        
        if actual_num_models < 1:
            if self.verbose:
                print(f"DPTukey: Cannot create {num_models} models with {n_samples} samples")
            return np.zeros((0, n_features))
        
        subset_size = n_samples // actual_num_models
        
        for i in range(actual_num_models):
            # Create subset for this model
            start_idx = i * subset_size
            end_idx = min(start_idx + subset_size, n_samples)
            
            if end_idx - start_idx < min_samples_per_model:
                break
                
            subset_x = x[start_idx:end_idx]
            subset_y = y[start_idx:end_idx]
            
            # Train simple logistic regression
            try:
                theta = self._train_single_model(subset_x, subset_y)
                models.append(theta)
            except Exception as e:
                if self.verbose:
                    print(f"DPTukey: Failed to train model {i}: {e}")
                continue
        
        return np.array(models) if models else np.zeros((0, n_features))
    
    def _train_single_model(self, x, y, max_iters=100, learning_rate=0.01, lambda_reg=1e-3):
        """Train a single logistic regression model using gradient descent."""
        n_samples, n_features = x.shape
        theta = np.zeros(n_features)
        
        # Convert labels to {-1, +1} if needed
        y_binary = np.where(y <= 0, -1, 1)
        
        for _ in range(max_iters):
            # Compute logistic regression gradient
            z = x.dot(theta)
            z_clipped = np.clip(y_binary * z, -500, 500)  # For numerical stability
            p = 1 / (1 + np.exp(-z_clipped))
            
            # Gradient of logistic loss + L2 regularization
            gradient = -x.T.dot(y_binary * (1 - p)) / n_samples + lambda_reg * theta
            
            # Update parameters
            theta = theta - learning_rate * gradient
            
            # Simple constraint to prevent explosion
            theta_norm = np.linalg.norm(theta)
            if theta_norm > 10.0:  # Arbitrary large bound
                theta = theta * (10.0 / theta_norm)
        
        return theta
    
    def _dp_tukey(self, models, epsilon, delta):
        """
        Run (epsilon, delta)-DP Tukey mechanism using models.
        
        Args:
            models: Feature vectors of non-private models (n_models × n_features)
            epsilon: Privacy parameter
            delta: Privacy parameter
            
        Returns:
            Selected model or zero vector if PTR fails
        """
        if models.shape[0] < 2:
            return np.zeros(models.shape[1] if models.shape[0] > 0 else 0)
        
        # Transpose for projections (features × models)
        projections = self._perturb_and_sort_matrix(models.T)
        max_depth = int(len(models) / 2)
        
        if max_depth < 1:
            return np.zeros(models.shape[1])
        
        # Compute log(volume_i) for i=1 to max_depth
        log_volumes = self._log_measure_geq_all_depths(projections)
        t = max(1, int(max_depth / 2))
        
        # Split privacy budget
        split_epsilon = epsilon / 2
        
        # PTR check
        distance = self._distance_to_unsafety(log_volumes, split_epsilon, delta, t, -1, t-1)
        threshold = np.log(1 / (2 * delta)) / split_epsilon
        
        # Add Laplace noise for PTR
        noisy_distance = distance + np.random.laplace(scale=1/split_epsilon)
        
        if not noisy_distance > threshold:
            if self.verbose:
                print("DPTukey: PTR check failed")
            return np.zeros(len(models[0]))
        
        # Sample depth using restricted exponential mechanism
        depth = self._restricted_racing_sample_depth(projections, split_epsilon, t)
        
        # Sample uniformly from the region of given depth
        return self._sample_exact(depth, projections[:, t:-t])
    
    def _perturb_and_sort_matrix(self, input_matrix):
        """Add small perturbation and sort each row."""
        d, m = input_matrix.shape
        perturbation_matrix = 1e-10 * np.random.rand(d, m)
        perturbed_matrix = input_matrix + perturbation_matrix
        return np.sort(perturbed_matrix, axis=1)
    
    def _log_measure_geq_all_depths(self, projections):
        """Compute log(volume) of regions of at least each depth."""
        if projections.shape[1] < 2:
            return np.array([])
        
        max_depth = int(np.ceil(projections.shape[1] / 2))
        if max_depth == 0:
            return np.array([])
        
        diff = np.flip(projections, axis=1) - projections
        
        # Take only up to max_depth columns
        diff_truncated = diff[:, :max_depth]
        
        # Handle potential numerical issues
        diff_truncated = np.maximum(diff_truncated, 1e-100)
        
        return np.sum(np.log(diff_truncated), axis=0)
    
    def _distance_to_unsafety(self, log_volumes, epsilon, delta, t, k_low, k_high):
        """Compute Hamming distance lower bound using PTR check."""
        if len(log_volumes) == 0 or k_high <= k_low:
            return k_low
        
        k = int((k_low + k_high) / 2)
        
        # Check bounds
        vol_idx = t - k - 2
        if vol_idx < 0 or vol_idx >= len(log_volumes):
            return k_low
        
        log_vol_ytk = log_volumes[vol_idx]
        
        start_idx = t + k + 1
        end_idx = len(log_volumes) - 1
        
        if start_idx >= end_idx or start_idx < 0:
            return k_low
        
        log_vols_ytk_gs = log_volumes[start_idx:end_idx]
        
        if len(log_vols_ytk_gs) == 0:
            return k_low
        
        log_epsilon_terms = (epsilon / 2) * np.arange(1, len(log_vols_ytk_gs) + 1)
        log_threshold = np.log(delta / (8 * np.exp(epsilon)))
        
        condition_values = (log_vol_ytk - log_vols_ytk_gs) - log_epsilon_terms
        
        if np.min(condition_values) <= log_threshold:
            if k_low >= k_high - 1:
                return k_high
            else:
                new_k_low = k_low + int((k_high - k_low) / 2)
                return self._distance_to_unsafety(log_volumes, epsilon, delta, t, new_k_low, k_high)
        
        if k_high > k_low + 1:
            new_k_high = k_high - int((k_high - k_low) / 2)
            return self._distance_to_unsafety(log_volumes, epsilon, delta, t, k_low, new_k_high)
        
        return k_low
    
    def _restricted_racing_sample_depth(self, projections, epsilon, restricted_depth):
        """Sample depth using exponential mechanism with racing sampling."""
        if projections.shape[1] < 2 * restricted_depth:
            return restricted_depth
        
        # Restrict projections to valid depth range
        start_idx = restricted_depth - 1
        end_idx = -(restricted_depth - 1) if restricted_depth > 1 else projections.shape[1]
        
        if start_idx >= end_idx:
            return restricted_depth
        
        restricted_projections = projections[:, start_idx:end_idx]
        
        atleast_volumes = np.exp(self._log_measure_geq_all_depths(restricted_projections))
        
        if len(atleast_volumes) == 0:
            return restricted_depth
        
        # Compute exact volumes
        measure_exact_all = atleast_volumes.copy()
        if len(measure_exact_all) > 1:
            measure_exact_all[:-1] = atleast_volumes[:-1] - atleast_volumes[1:]
        
        depths = np.arange(restricted_depth, restricted_depth + len(atleast_volumes))
        
        # Handle numerical issues
        measure_exact_all = np.maximum(measure_exact_all, 1e-100)
        log_terms = np.log(measure_exact_all) + epsilon * depths
        
        # Racing sample
        sampled_idx = self._racing_sample(log_terms)
        return depths[sampled_idx]
    
    def _racing_sample(self, log_terms):
        """Numerically stable racing sampling."""
        if len(log_terms) == 0:
            return 0
        if len(log_terms) == 1:
            return 0
        
        uniform_samples = np.random.uniform(size=log_terms.shape)
        # Avoid log(0)
        uniform_samples = np.maximum(uniform_samples, 1e-100)
        
        log_log_terms = np.log(np.maximum(-np.log(uniform_samples), 1e-100))
        racing_values = log_log_terms - log_terms
        
        return np.argmin(racing_values)
    
    def _sample_exact(self, depth, projections):
        """Sample uniformly from the Tukey region of given depth."""
        d, m = projections.shape
        result = np.zeros(d)
        
        for i in range(d):
            result[i] = self._sample_exact_1d(depth, projections[i])
        
        return result
    
    def _sample_exact_1d(self, depth, projection):
        """Sample point of exactly given depth from 1D projection."""
        n = len(projection)
        
        # Ensure valid depth
        max_valid_depth = n // 2
        if depth < 1 or depth > max_valid_depth:
            # Return midpoint if invalid depth
            return (projection[0] + projection[-1]) / 2
        
        # Get the four boundary points
        left_low = projection[depth-1]
        left_high = projection[depth] if depth < n else projection[-1]
        right_low = projection[-(depth+1)] if depth+1 <= n else projection[0]
        right_high = projection[-depth]
        
        # Ensure proper ordering
        if left_high < left_low:
            left_high = left_low
        if right_high < right_low:
            right_high = right_low
        
        measure_left = max(0, left_high - left_low)
        measure_right = max(0, right_high - right_low)
        total_measure = measure_left + measure_right
        
        if total_measure <= 0:
            return (left_low + right_high) / 2
        
        # Sample from left or right region
        if np.random.uniform() < measure_left / total_measure:
            return left_low + np.random.uniform() * measure_left
        else:
            return right_low + np.random.uniform() * measure_right
    
    @property
    def name(self):
        return "DP-Tukey"
