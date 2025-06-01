import numpy as np
import math
import sklearn.linear_model # For LogisticRegression
from common.common import Algorithm # Assuming Algorithm is your base class
from common.clipping import clip_rows # If you use it for L_data_clipping
# from common.constraints import constrain_l2_norm # Not typically used by TukeyEM directly

# --- TukeyEM Core Functions (Adapted from Google Research & Paper) ---
# (Paste all the refined TukeyEM helper functions here, ensuring they take 'rng' where needed)
# - perturb_and_sort_matrix(input_matrix, rng_pert)
# - log_measure_geq_all_depths(projections)
# - racing_sample(log_terms, rng)
# - restricted_racing_sample_depth(projections, epsilon_exp_mech, t_min_depth_for_sampling, rng)
# - distance_to_unsafety(log_V_series_ptr, epsilon_ptr_half, delta_for_threshold, t_ptr_center_depth, rng_laplace_for_ptr_noise)
# - log_measure_geq_all_dims(depth_1_idx, projections)
# - sample_geq_1d(depth_1_idx, projection_1d, rng)
# - sample_exact_1d(depth_1_idx, projection_1d, rng)
# - sample_exact(depth_1_idx, projections, rng)
# - multiple_logistic_regressions(features, labels, num_models, rng_main_shuffle, ...)
# - dp_tukey_mechanism(initial_models_coeffs, epsilon_total, delta_total, rng_for_mechanism, ...)

# --- Example definitions (ensure these are the robust versions we discussed) ---
def perturb_and_sort_matrix(input_matrix, rng_pert):
    input_matrix = np.asarray(input_matrix)
    if input_matrix.size == 0: return input_matrix
    d, m = input_matrix.shape
    if m == 0: return input_matrix
    perturbation_matrix = 1e-10 * rng_pert.random((d, m))
    perturbed_matrix = input_matrix + perturbation_matrix
    return np.sort(perturbed_matrix, axis=1)

def log_measure_geq_all_depths(projections):
    projections = np.asarray(projections)
    if projections.ndim != 2 or projections.shape[0] == 0 or projections.shape[1] < 2:
        return np.array([-np.inf])
    d_features, num_models = projections.shape
    max_depth_k = num_models // 2
    if max_depth_k == 0: return np.array([-np.inf])
    log_volumes_V_geq_k = np.full(max_depth_k, -np.inf)
    for k_depth_1_indexed in range(1, max_depth_k + 1):
        widths_for_depth_k = projections[:, (num_models - k_depth_1_indexed)] - projections[:, k_depth_1_indexed - 1]
        valid_widths = widths_for_depth_k[widths_for_depth_k > 1e-100]
        if len(valid_widths) < d_features or np.any(valid_widths <= 0):
            log_volumes_V_geq_k[k_depth_1_indexed - 1] = -np.inf
        else:
            log_volumes_V_geq_k[k_depth_1_indexed - 1] = np.sum(np.log(valid_widths))
    return log_volumes_V_geq_k

def racing_sample(log_terms, rng):
    if not isinstance(log_terms, np.ndarray): log_terms = np.array(log_terms)
    if len(log_terms) == 0: raise ValueError("racing_sample called with empty log_terms")
    if np.all(np.isneginf(log_terms)): return rng.choice(len(log_terms))
    max_log_term = np.max(log_terms[np.isfinite(log_terms)]) if np.any(np.isfinite(log_terms)) else 0
    stable_log_terms = log_terms - max_log_term
    random_variates = rng.uniform(size=stable_log_terms.shape)
    log_of_log_uniform = np.log(np.maximum(1e-100, np.log(1.0 / np.maximum(1e-100, random_variates))))
    return np.argmin(log_of_log_uniform - stable_log_terms)

def restricted_racing_sample_depth(projections, epsilon_exp_mech, t_min_depth_for_sampling, rng):
    num_models = projections.shape[1]
    max_possible_depth_m_div_2 = num_models // 2
    if max_possible_depth_m_div_2 == 0: return 1 if num_models >= 1 else 0
    t_min_depth = max(1, int(t_min_depth_for_sampling))
    if t_min_depth > max_possible_depth_m_div_2: t_min_depth = max_possible_depth_m_div_2
    depths_to_sample_from = np.arange(t_min_depth, max_possible_depth_m_div_2 + 1)
    if len(depths_to_sample_from) == 0: return t_min_depth
    log_V_series = log_measure_geq_all_depths(projections)
    log_score_terms = []
    for i_depth in depths_to_sample_from:
        idx_V_i = i_depth - 1
        idx_V_i_plus_1 = i_depth
        log_V_i_val = log_V_series[idx_V_i] if idx_V_i < len(log_V_series) else -np.inf
        log_V_i_plus_1_val = log_V_series[idx_V_i_plus_1] if idx_V_i_plus_1 < len(log_V_series) else -np.inf
        W_i_volume = max(0, np.exp(log_V_i_val) - np.exp(log_V_i_plus_1_val))
        current_log_score = -np.inf
        if W_i_volume > 1e-100: current_log_score = np.log(W_i_volume) + epsilon_exp_mech * i_depth
        log_score_terms.append(current_log_score)
    if not log_score_terms: return t_min_depth
    log_score_terms = np.array(log_score_terms)
    sampled_offset_in_range = racing_sample(log_score_terms, rng)
    return depths_to_sample_from[sampled_offset_in_range]

def distance_to_unsafety(log_V_series_ptr, epsilon_ptr_half, delta_for_threshold, t_ptr_center_depth, rng_laplace_for_ptr_noise): # Added rng for laplace
    best_k_found = -1
    for k_val in range(t_ptr_center_depth - 1, -1, -1):
        depth_num = t_ptr_center_depth - k_val - 1
        if depth_num < 1: continue
        log_V_numerator_val = log_V_series_ptr[depth_num - 1] if (depth_num - 1) < len(log_V_series_ptr) else -np.inf
        condition_met_for_k = False
        for g_val in range(1, len(log_V_series_ptr) + 2):
            depth_den = t_ptr_center_depth + k_val + g_val + 1
            log_V_denominator_val = log_V_series_ptr[depth_den - 1] if (depth_den -1) < len(log_V_series_ptr) and depth_den >=1 else -np.inf
            current_log_ratio = -np.inf
            if not np.isneginf(log_V_numerator_val):
                if np.isneginf(log_V_denominator_val): current_log_ratio = np.inf
                else: current_log_ratio = log_V_numerator_val - log_V_denominator_val
            if current_log_ratio - (epsilon_ptr_half * g_val / 2.0) <= np.log(max(delta_for_threshold, 1e-100)):
                best_k_found = k_val; condition_met_for_k = True; break
        if condition_met_for_k: return best_k_found
    return best_k_found

def log_measure_geq_all_dims(depth_1_idx, projections):
    projections = np.asarray(projections)
    if projections.ndim != 2 or depth_1_idx <= 0 or projections.shape[1] < 2*depth_1_idx :
        return np.full(projections.shape[0], -np.inf) if projections.ndim == 2 else np.array([-np.inf])
    widths = projections[:, -depth_1_idx] - projections[:, depth_1_idx - 1]
    return np.log(np.maximum(widths, 1e-100))

def sample_geq_1d(depth_1_idx, projection_1d, rng):
    num_pts = len(projection_1d)
    safe_depth = min(max(1, depth_1_idx), num_pts // 2) if num_pts >=2 else 1
    if num_pts < 2 * safe_depth and num_pts > 0 : return projection_1d[0]
    if num_pts == 0: return 0.0
    low = projection_1d[safe_depth-1]; high = projection_1d[-safe_depth]
    if high < low: high = low
    return rng.uniform(low, high)

def sample_exact_1d(depth_1_idx, projection_1d, rng):
    num_pts = len(projection_1d)
    if not (1 <= depth_1_idx <= num_pts // 2) or num_pts < 2:
        print(f"Warning (TukeyEM): sample_exact_1d with invalid depth {depth_1_idx} for {num_pts} points.")
        return (projection_1d[0] + projection_1d[-1])/2 if num_pts > 1 else (projection_1d[0] if num_pts==1 else 0.0)
    idx_left_low = depth_1_idx - 1; idx_left_high = depth_1_idx 
    idx_right_low = -(depth_1_idx + 1); idx_right_high = -depth_1_idx     
    left_low_val = projection_1d[idx_left_low]
    left_high_val = projection_1d[idx_left_high] if idx_left_high < num_pts else left_low_val 
    right_low_val = projection_1d[idx_right_low if idx_right_low >= -num_pts else -num_pts]
    right_high_val = projection_1d[idx_right_high]
    measure_left = max(0, left_high_val - left_low_val); measure_right = max(0, right_high_val - right_low_val)
    total_measure = measure_left + measure_right
    if total_measure <= 1e-100: return (left_low_val + right_high_val) / 2.0
    if rng.uniform() < measure_left / total_measure:
        return rng.uniform(left_low_val, left_high_val) if measure_left > 0 else left_low_val
    else:
        return rng.uniform(right_low_val, right_high_val) if measure_right > 0 else right_low_val

def sample_exact(depth_1_idx, projections, rng):
    projections = np.asarray(projections)
    d_features, m_models = projections.shape
    if not (1 <= depth_1_idx <= m_models // 2):
        raise ValueError(f"Sample_exact: Depth {depth_1_idx} out of range [1, {m_models//2}] for {m_models} models.")
    log_lengths_geq_depth_ip1 = log_measure_geq_all_dims(depth_1_idx + 1, projections)
    log_lengths_geq_depth_i = log_measure_geq_all_dims(depth_1_idx, projections)
    exact_lengths_W_ji = np.maximum(0, np.exp(log_lengths_geq_depth_i) - np.exp(log_lengths_geq_depth_ip1))
    log_exact_lengths_W_ji = np.log(np.maximum(exact_lengths_W_ji, 1e-100))
    log_term_volumes_C_jstar_i = np.zeros(d_features)
    for j_star in range(d_features):
        log_V_prefix = np.sum(log_lengths_geq_depth_ip1[:j_star])
        log_W_jstar_i_val = log_exact_lengths_W_ji[j_star]
        log_V_suffix = np.sum(log_lengths_geq_depth_i[j_star+1:])
        log_term_volumes_C_jstar_i[j_star] = log_V_prefix + log_W_jstar_i_val + log_V_suffix
    sampled_j_star_dim_idx = racing_sample(log_term_volumes_C_jstar_i, rng)
    sample_beta_vector = np.zeros(d_features)
    for j_current_dim in range(d_features):
        current_1d_projection = projections[j_current_dim, :]
        depth_for_geq_ip1 = min(depth_1_idx + 1, len(current_1d_projection) // 2)
        depth_for_geq_i = min(depth_1_idx, len(current_1d_projection) // 2)
        if depth_for_geq_ip1 == 0: depth_for_geq_ip1 = 1 
        if depth_for_geq_i == 0: depth_for_geq_i = 1
        if j_current_dim < sampled_j_star_dim_idx:
            sample_beta_vector[j_current_dim] = sample_geq_1d(depth_for_geq_ip1, current_1d_projection, rng)
        elif j_current_dim == sampled_j_star_dim_idx:
            sample_beta_vector[j_current_dim] = sample_exact_1d(depth_1_idx, current_1d_projection, rng)
        else:
            sample_beta_vector[j_current_dim] = sample_geq_1d(depth_for_geq_i, current_1d_projection, rng)
    return sample_beta_vector

def multiple_logistic_regressions(features, labels, num_models_requested, rng_main_shuffle, 
                                  fit_intercept_flag=False, C_reg=1.0, solver='liblinear',
                                  min_samples_per_model_heuristic=10):
    features = np.asarray(features); labels = np.asarray(labels)
    (n_samples_total, d_features_proc) = features.shape
    if n_samples_total == 0: return np.zeros((0, d_features_proc))
    if d_features_proc == 0 and not fit_intercept_flag: return np.zeros((num_models_requested, 0))

    min_samples_for_one_lr = max(min_samples_per_model_heuristic, d_features_proc + 1 if d_features_proc > 0 else 2)
    actual_num_models = num_models_requested
    if num_models_requested <= 0: actual_num_models = 1
    
    max_possible_m = n_samples_total // min_samples_for_one_lr
    if actual_num_models > max_possible_m:
        original_m_req = actual_num_models
        actual_num_models = max(1, max_possible_m)
        if actual_num_models == 0 : return np.zeros((0, d_features_proc))
        print(f"Warning (TukeyEM): num_models reduced from {original_m_req} to {actual_num_models} (need {min_samples_for_one_lr} samples per model).")

    if actual_num_models == 0: return np.zeros((0, d_features_proc)) # Should be caught above
    batch_size = n_samples_total // actual_num_models
    num_samples_to_use = batch_size * actual_num_models 
    if num_samples_to_use == 0: return np.zeros((0, d_features_proc))
        
    order = rng_main_shuffle.permutation(n_samples_total)
    shuffled_features = features[order[:num_samples_to_use]]
    shuffled_labels = labels[order[:num_samples_to_use]]
    
    models_coeffs_list = []
    for i in range(actual_num_models):
        start_idx = i * batch_size; end_idx = start_idx + batch_size
        features_batch = shuffled_features[start_idx:end_idx, :]
        labels_batch = shuffled_labels[start_idx:end_idx]
        
        if len(np.unique(labels_batch)) < 2: print(f"Warn (TukeyEM): Subset {i} has only one class. Skipping."); continue
        if features_batch.shape[0] < min_samples_for_one_lr : print(f"Warn (TukeyEM): Subset {i} too small. Skipping."); continue

        try:
            log_reg_model = sklearn.linear_model.LogisticRegression(
                fit_intercept=fit_intercept_flag, C=C_reg, solver=solver, max_iter=200, 
                random_state=rng_main_shuffle.integers(1000000) )
            log_reg_model.fit(features_batch, labels_batch)
            current_coeffs = log_reg_model.coef_.flatten()
            if fit_intercept_flag and hasattr(log_reg_model, 'intercept_') and d_features_proc == features_batch.shape[1] + 1 : # If sklearn added intercept
                current_coeffs = np.hstack((log_reg_model.intercept_, current_coeffs))
            
            if len(current_coeffs) == d_features_proc: models_coeffs_list.append(current_coeffs)
            else: print(f"Warn (TukeyEM): Coeff shape mismatch {len(current_coeffs)} vs {d_features_proc}. Skipping.")
        except Exception as e: print(f"Warn (TukeyEM): LogReg error on subset {i}: {e}. Skipping.")
            
    return np.array(models_coeffs_list) if models_coeffs_list else np.zeros((0, d_features_proc))

def dp_tukey_mechanism(
    initial_models_coeffs, epsilon_total, delta_total, rng_for_mechanism,
    ptr_t_depth_center_divisor=4, exp_mech_t_depth_min_divisor=4 ):
    if initial_models_coeffs.ndim != 2 or initial_models_coeffs.shape[0] == 0:
        return np.zeros(initial_models_coeffs.shape[1] if initial_models_coeffs.ndim == 2 and initial_models_coeffs.shape[0] > 0 else 0)

    m_models, d_features = initial_models_coeffs.shape
    if m_models < 2: return np.mean(initial_models_coeffs, axis=0) if m_models == 1 else np.zeros(d_features)
    if d_features == 0: return np.array([])

    projections = perturb_and_sort_matrix(initial_models_coeffs.T, rng_for_mechanism)
    log_V_series = log_measure_geq_all_depths(projections)
    if np.all(np.isneginf(log_V_series)): return np.zeros(d_features)

    max_depth_possible = m_models // 2
    if max_depth_possible == 0: return np.zeros(d_features)

    epsilon_ptr_budget = epsilon_total / 2.0; epsilon_exp_mech_budget = epsilon_total / 2.0
    t_ptr_center_val = max(1, max_depth_possible // ptr_t_depth_center_divisor)
    
    k_dist_lower_bound = distance_to_unsafety(log_V_series, epsilon_ptr_budget, delta_total, t_ptr_center_val, rng_for_mechanism) #rng for laplace if used inside
    
    # Laplace mechanism for PTR: sensitivity of k_dist is 1. Scale is 1/epsilon_ptr_budget.
    ptr_noise_scale = 1.0 / max(epsilon_ptr_budget, 1e-100) # Avoid division by zero
    noisy_distance_val = k_dist_lower_bound + rng_for_mechanism.laplace(loc=0, scale=ptr_noise_scale)
    ptr_comparison_threshold = np.log(1.0 / max(2 * delta_total, 1e-100)) / max(epsilon_ptr_budget, 1e-100)

    if not (noisy_distance_val > ptr_comparison_threshold):
        print("Warn (TukeyEM dp_mech): PTR check failed. Returning zero model.")
        return np.zeros(d_features)

    t_exp_mech_min_depth_val = max(1, m_models // exp_mech_t_depth_min_divisor)
    if t_exp_mech_min_depth_val > max_depth_possible : t_exp_mech_min_depth_val = max_depth_possible
    
    sampled_final_depth = restricted_racing_sample_depth(projections, epsilon_exp_mech_budget, t_exp_mech_min_depth_val, rng_for_mechanism)
    if sampled_final_depth == 0 and m_models > 0: return np.zeros(d_features)

    return sample_exact(sampled_final_depth, projections, rng_for_mechanism)

# --- Wrapper Class for TukeyEM (for Logistic Regression coefficients) ---
class TukeyEMLR(Algorithm):
    def run_classification(self, x, y, epsilon, delta, m, 
                           random_state=None, L_data_clipping=None, 
                           logreg_C=1.0, logreg_solver='liblinear', logreg_fit_intercept=False,
                           ptr_t_divisor=4, exp_mech_t_divisor=4, **kwargs):
        rng = np.random.default_rng(random_state) 
        current_x_features = np.asarray(x.toarray() if hasattr(x, "toarray") else x) 
        current_y_labels = np.asarray(y)

        d_features_expected = current_x_features.shape[1]
        if logreg_fit_intercept and not self._has_intercept_column(current_x_features):
             # If we tell sklearn to fit intercept, it expects data *without* an explicit intercept column.
             # The returned coef_ will be d_features, and intercept_ will be separate.
             # Our multiple_logistic_regressions needs to handle this to return d_features+1 coeffs.
             # For now, assume logreg_fit_intercept=False means X has intercept.
             pass


        if current_x_features.shape[0] == 0:
            return np.zeros(d_features_expected), L_data_clipping, m # Return m for logging
        if d_features_expected == 0 and not logreg_fit_intercept:
             return np.array([]), L_data_clipping, m

        x_to_use = current_x_features
        if L_data_clipping is not None:
            x_to_use = clip_rows(current_x_features, 2, L_data_clipping)

        initial_logistic_coeffs = multiple_logistic_regressions(
            x_to_use, current_y_labels, m, rng, 
            fit_intercept_flag=logreg_fit_intercept, C_reg=logreg_C, solver=logreg_solver
        )
        
        if initial_logistic_coeffs.ndim != 2 or initial_logistic_coeffs.shape[0] == 0:
            print(f"Warn (TukeyEMLR): No initial logistic models generated (shape: {initial_logistic_coeffs.shape}). Returning zeros.")
            return np.zeros(d_features_expected), L_data_clipping, m

        final_coeffs = dp_tukey_mechanism(
            initial_logistic_coeffs, epsilon, delta, rng, 
            ptr_t_depth_center_divisor=ptr_t_divisor,
            exp_mech_t_depth_min_divisor=exp_mech_t_divisor
        )
        return final_coeffs, L_data_clipping, m 

    def _has_intercept_column(self, X): # Helper if needed
        return np.all(X[:, 0] == 1) if X.shape[1] > 0 else False

    @property
    def name(self):
        return "TukeyEM-LR"