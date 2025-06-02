import os
import sys
import csv
import math
import numpy as np
import sklearn.linear_model
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from time import process_time
from multiprocessing import Pool 
from itertools import product
from Algorithms.dpsgd_lr import DPSGDLR
from Algorithms.amp_lr import ApproximateMinimaPerturbationLR 
from Algorithms.tukeyEM_lr import TukeyEMLR

##########################################################################
# --- Global Constants and Configuration ---
##########################################################################
DATASET_LOCATION = '/Users/anass/Desktop/Thesis/Code/Thesis_code/datasets/data'
AVAILABLE_DATASETS = ['california', 'diamonds', 'traffic', 'adult'] 

NUM_REPEATS = 3
CORES = 4 
EPSILON_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
MAIN_SEED = 42
MIN_SUBSET_SIZE = 50 
MODEL_TRAIN_TIMEOUT = 300 
MODEL_EVAL_TIMEOUT = 60

##########################################################################
# --- Helper Functions ---
##########################################################################
def build_binary_labels(y, positive_class_label=1):
    y_arr = np.asarray(y).ravel()
    unique_y = set(np.unique(y_arr))
    if unique_y.issubset({-1, 1}):
        return np.where(y_arr == 1, 1, -1).astype(int)
    return np.where(y_arr == positive_class_label, 1, -1).astype(int)

def progress_bar(pct):
    i = int(pct)
    sys.stdout.write('\r')
    sys.stdout.write("\033[K[ %-20s ] %d%%" % ('=' * int(i / 5), i))
    sys.stdout.flush()

def create_directory(directory_name):
    try:
        os.makedirs(directory_name, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {directory_name}: {e}")

def dict_product(param_dict):
    keys = param_dict.keys()
    values = param_dict.values()
    for combination in product(*values):
        yield dict(zip(keys, combination))

##########################################################################
# --- Dataset Loading ---
##########################################################################
def load_dataset(dataset_name):
    if dataset_name not in AVAILABLE_DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' not available. Choose from: {AVAILABLE_DATASETS}")
    
    x_path = os.path.join(DATASET_LOCATION, f"{dataset_name}_processed_x.npy")
    y_path = os.path.join(DATASET_LOCATION, f"{dataset_name}_processed_y.npy")
    
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        raise FileNotFoundError(f"Dataset files not found for '{dataset_name}' in '{DATASET_LOCATION}'. Ensure processor was run.")
    
    X = np.load(x_path).astype(float)
    y_original = np.load(y_path).astype(int).ravel()
    
    print(f"Dataset '{dataset_name}': Loaded X shape={X.shape}, Original y shape={y_original.shape}")
    unique_labels_orig, counts_orig = np.unique(y_original, return_counts=True)
    print(f"  Original class distribution: {dict(zip(unique_labels_orig, counts_orig))}")

    if 1 in unique_labels_orig:
        positive_class_val = 1
    elif 0 in unique_labels_orig and len(unique_labels_orig) == 2:
        positive_class_val = np.max(unique_labels_orig)
    else:
        positive_class_val = np.max(unique_labels_orig) if unique_labels_orig.size > 0 else 1
    
    y_binary = build_binary_labels(y_original, positive_class_label=positive_class_val)
    unique_labels_binary, counts_binary = np.unique(y_binary, return_counts=True)
    print(f"  Converted y to binary {-1, 1} (positive class original label: {positive_class_val}). New distribution: {dict(zip(unique_labels_binary, counts_binary))}")
    
    return X, y_binary

##########################################################################
# --- Algorithm Evaluation Functions ---
##########################################################################
def predict_binary(theta, X):
    if theta is None or len(theta) == 0:
        return np.ones(X.shape[0], dtype=int) * -1 
    scores = X.dot(theta)
    return np.where(scores >= 0, 1, -1).astype(int)

def evaluate_model_accuracy(theta, X, y_true):
    if theta is None: return 0.0
    y_pred = predict_binary(theta, X)
    return np.mean(y_pred == y_true)

def count_errors(theta, X, y_true):
    if theta is None: return X.shape[0] if X.shape[0] > 0 else 0
    if X.shape[0] == 0: return 0
    y_pred = predict_binary(theta, X)
    return np.sum(y_pred != y_true)

##########################################################################
# --- Training Task for Parallel Execution ---
##########################################################################
def train_model_task(args):
    alg_class = args['alg_class']
    X_train_subset, y_train_subset = args['X_train_subset'], args['y_train_subset']
    epsilon_for_training, delta = args['epsilon_for_training'], args['delta']
    hyperparams, random_state = args['hyperparams'], args['random_state']
    
    try:
        # Assuming alg_class() constructor doesn't require args or accepts verbose=False by default
        # If your corrected AMP class now takes verbose in __init__, you might pass it here:
        # algorithm = alg_class(verbose=False) 
        algorithm = alg_class() 
        theta, _, _ = algorithm.run_classification(
            x=X_train_subset, y=y_train_subset, 
            epsilon=epsilon_for_training, delta=delta,
            random_state=random_state, **hyperparams)
        success = theta is not None and not np.allclose(theta, 0.0, atol=1e-12)
        return {'theta': theta if success else None, 'hyperparams': hyperparams, 'success': success}
    except Exception:
        return {'theta': None, 'hyperparams': hyperparams, 'success': False}

def evaluate_candidate_task(args):
    theta, X_val, y_val = args['theta'], args['X_val'], args['y_val']
    hyperparams = args['hyperparams']
    errors = count_errors(theta, X_val, y_val)
    return {'errors': errors, 'hyperparams': hyperparams, 'theta': theta}

##########################################################################
# --- Hyperparameter Configuration ---
##########################################################################
def get_hyperparameter_configs(dataset_name, n_samples):
    """
    FIXED hyperparameter configuration with much more conservative TukeyEM settings.
    
    Key changes for TukeyEM:
    - Much smaller m values (5-15 instead of 20-50)
    - More lenient PTR parameters (divisors 8-16 instead of 2-4)
    - Simplified C values to reduce complexity
    """
    base_lambda = [1e-4, 1e-3, 1e-2]
    base_epochs = [20, 50, 100]
    base_batch_sizes = [32, 64]
    base_L_dpsgd = [0.5, 1.0, 2.0]
    
    # SIMPLIFIED AMP parameters (auto eps_frac calculation)
    amp_L_grad_bound = [1.0, 2.0]              
    amp_l2_constraint = [None, 5.0]            
    amp_eps_out_frac = [0.01, 0.05]            
    
    # MUCH MORE CONSERVATIVE TukeyEM parameters
    if dataset_name == "california":
        dpsgd_epochs_specific = base_epochs
        tukey_m_specific = [5, 8, 12]  # Much smaller, was [20, 50]
    elif dataset_name == "diamonds":
        dpsgd_epochs_specific = base_epochs[:2]
        tukey_m_specific = [8, 12, 15]  # Much smaller, was [50, 100]
    elif dataset_name == "traffic":
        dpsgd_epochs_specific = base_epochs
        tukey_m_specific = [3, 5, 8]   # Much smaller, was [20, 50]
    else: 
        dpsgd_epochs_specific = base_epochs[:2]
        tukey_m_specific = [5, 8]      # Conservative fallback
    
    return {
        'DPSGD': {
            'class': DPSGDLR,
            'params': {
                'lambda_param': base_lambda, 
                'num_epochs': dpsgd_epochs_specific,
                'batch_size': base_batch_sizes, 
                'L': base_L_dpsgd
            }
        },
        'AMP': {
            'class': ApproximateMinimaPerturbationLR,
            'params': {
                'L': amp_L_grad_bound,                    
                'l2_constraint': amp_l2_constraint,       
                'eps_out_frac': amp_eps_out_frac,         
            }
        },
        'TukeyEM': {
            'class': TukeyEMLR,
            'params': {
                'm': tukey_m_specific,                    # Much smaller values
                'L_data_clipping': [None],                # Keep simple
                'logreg_C': [1.0],                        # Simplified, was [0.1, 1.0, 10.0]
                'logreg_fit_intercept': [False],          # Keep consistent
                'ptr_t_divisor': [8, 12, 16],             # Much more lenient, was [2, 4]
                'exp_mech_t_divisor': [8, 12, 16],        # Much more lenient, was [2, 4]
                'logreg_solver': ['liblinear'],           # Keep consistent
                'debug': [False]                          # Can enable for debugging
            }
        }
    }

##########################################################################
# --- Private Hyperparameter Tuning (Algorithm 3 Style) ---
##########################################################################
def private_hyperparameter_tuning(alg_class, param_grid, 
                                  X_tuning_data, y_tuning_data, 
                                  epsilon_tuning_phase_budget, delta, 
                                  random_state_base):
    param_combinations = list(dict_product(param_grid))
    n_candidates = len(param_combinations)

    print(f"Starting private hyperparameter tuning with {n_candidates} configurations.")
    if n_candidates == 0:
        print("  No hyperparameter configurations. Fallback: using empty params, max errors.")
        return {}, (X_tuning_data.shape[0] if X_tuning_data.shape[0] > 0 else 0)

    epsilon_for_all_candidate_trainings = epsilon_tuning_phase_budget / 2.0
    epsilon_for_selection_mechanism = epsilon_tuning_phase_budget / 2.0
    print(f"  Budget for tuning phase (epsilon={epsilon_tuning_phase_budget:.4f}):")
    print(f"    - For ALL candidate trainings (Parallel Comp. epsilon={epsilon_for_all_candidate_trainings:.4f})")
    print(f"    - For Exponential Mechanism selection (epsilon={epsilon_for_selection_mechanism:.4f})")

    n_tuning_samples = X_tuning_data.shape[0]
    # Ensure n_candidates + 1 is not zero before division
    if n_candidates < 0: # Should not happen with list conversion, but as a safeguard
        print("  Error: Negative number of candidates. Fallback: using first hyperparameter set.")
        return param_combinations[0] if param_combinations else {}, n_tuning_samples

    num_splits_required = n_candidates + 1
    if num_splits_required == 0: # Only if n_candidates = -1, highly unlikely
         print("  Error: (n_candidates + 1) is zero. Cannot split data. Fallback to first param.")
         return param_combinations[0] if param_combinations else {}, n_tuning_samples

    subset_size_for_training = n_tuning_samples // num_splits_required

    if subset_size_for_training < MIN_SUBSET_SIZE:
        print(f"  Insufficient data for {n_candidates} candidates + validation.")
        print(f"    Required per training split: {MIN_SUBSET_SIZE}, available: {subset_size_for_training} (from {n_tuning_samples} tuning samples / {num_splits_required} total splits).")
        print(f"    Fallback: using first hyperparameter set from grid (if any).")
        return param_combinations[0] if param_combinations else {}, n_tuning_samples

    rng_data_split = np.random.default_rng(random_state_base)
    permuted_indices = rng_data_split.permutation(n_tuning_samples)
    candidate_training_subsets_X, candidate_training_subsets_y = [], []
    for i in range(n_candidates):
        start_idx, end_idx = i * subset_size_for_training, (i + 1) * subset_size_for_training
        subset_indices = permuted_indices[start_idx:end_idx]
        candidate_training_subsets_X.append(X_tuning_data[subset_indices])
        candidate_training_subsets_y.append(y_tuning_data[subset_indices])
    
    validation_indices = permuted_indices[n_candidates * subset_size_for_training:]
    X_val, y_val = X_tuning_data[validation_indices], y_tuning_data[validation_indices]

    if X_val.shape[0] == 0:
        print("  No validation data available after splitting. Fallback: using first hyperparameter set.")
        return param_combinations[0] if param_combinations else {}, n_tuning_samples

    print(f"  Data split: {n_candidates} training subsets of ~{subset_size_for_training} samples. Validation set size: {X_val.shape[0]}.")
    print(f"  Training {n_candidates} candidate models...")
    train_tasks_args = [{
        'alg_class': alg_class, 'X_train_subset': candidate_training_subsets_X[i],
        'y_train_subset': candidate_training_subsets_y[i],
        'epsilon_for_training': epsilon_for_all_candidate_trainings, 'delta': delta,
        'hyperparams': params, 'random_state': random_state_base + i + 1
    } for i, params in enumerate(param_combinations)]

    trained_candidate_results = []
    with Pool(CORES) as pool:
        async_results = [pool.apply_async(train_model_task, (task_args,)) for task_args in train_tasks_args]
        for i, async_res in enumerate(async_results):
            progress_bar((i + 1) * 100 / n_candidates)
            try:
                trained_candidate_results.append(async_res.get(timeout=MODEL_TRAIN_TIMEOUT))
            except Exception as e:
                print(f"\n  Timeout/Error training candidate (params: {param_combinations[i]}): {type(e).__name__}")
                trained_candidate_results.append({'theta': None, 'hyperparams': param_combinations[i], 'success': False})
    sys.stdout.write('\r\033[K') 
    print("  Candidate model training complete.")
    
    print(f"  Evaluating {n_candidates} candidates on validation set...")
    eval_tasks_args = [{'theta': res['theta'], 'X_val': X_val, 'y_val': y_val,
                       'hyperparams': res['hyperparams']} for res in trained_candidate_results]
    validation_eval_results = []
    use_pool_for_eval = n_candidates > CORES * 2 
    if use_pool_for_eval and X_val.shape[0] > 0 :
        with Pool(CORES) as pool:
            async_results_eval = [pool.apply_async(evaluate_candidate_task, (task_args,)) for task_args in eval_tasks_args]
            for i, async_res_eval in enumerate(async_results_eval):
                progress_bar((i + 1) * 100 / n_candidates)
                try:
                    validation_eval_results.append(async_res_eval.get(timeout=MODEL_EVAL_TIMEOUT))
                except Exception as e:
                    print(f"\n  Error evaluating candidate: {type(e).__name__}")
                    validation_eval_results.append({'errors': X_val.shape[0], 'hyperparams': eval_tasks_args[i]['hyperparams'], 'theta': None})
        sys.stdout.write('\r\033[K')
        print("  Candidate evaluation complete (parallel).")
    elif X_val.shape[0] > 0:
        for i, task_args in enumerate(eval_tasks_args):
            progress_bar((i + 1) * 100 / n_candidates)
            validation_eval_results.append(evaluate_candidate_task(task_args))
        sys.stdout.write('\r\033[K')
        print("  Candidate evaluation complete (sequential).")
    else: # Should not happen if X_val.shape[0] check passed earlier
        validation_eval_results = [{'errors': 0, 'hyperparams': args['hyperparams'], 'theta': args['theta']} for args in eval_tasks_args]
        print("  Skipped candidate evaluation (no validation data).")

    error_counts_list = [res['errors'] for res in validation_eval_results]
    
    print(f"  Applying Exponential Mechanism for selection (epsilon={epsilon_for_selection_mechanism:.4f})...")
    scores = -np.array(error_counts_list, dtype=float)
    sensitivity_delta_s = 1.0 
    
    if scores.size == 0:
        print("  No scores for Exponential Mechanism. Fallback: using first hyperparameter set.")
        return param_combinations[0] if param_combinations else {}, (X_tuning_data.shape[0] if X_tuning_data.shape[0] > 0 else 0)

    max_score = np.max(scores)
    if np.isinf(max_score): 
        max_score = -float(X_val.shape[0] if X_val.shape[0] > 0 else 1.0)
    stable_scores = scores - max_score 
    stable_scores[np.isneginf(scores)] = -float(X_val.shape[0] if X_val.shape[0] > 0 else 1.0) - max_score 
    exp_weights = np.exp(epsilon_for_selection_mechanism * stable_scores / (2 * sensitivity_delta_s))

    if np.sum(exp_weights) == 0 or not np.all(np.isfinite(exp_weights)):
        print("  All exponential weights zero/non-finite. Using uniform probability for selection.")
        probabilities = np.ones(n_candidates) / n_candidates if n_candidates > 0 else np.array([])
    else:
        probabilities = exp_weights / np.sum(exp_weights)

    if probabilities.size == 0: 
         print("  No candidates to select from. Fallback: using first hyperparameter set.")
         return param_combinations[0] if param_combinations else {}, (X_tuning_data.shape[0] if X_tuning_data.shape[0] > 0 else 0)

    rng_selection = np.random.default_rng(random_state_base + n_candidates + 101)
    selected_idx = rng_selection.choice(n_candidates, p=probabilities)
    selected_hyperparams = param_combinations[selected_idx]
    selected_model_validation_errors = error_counts_list[selected_idx]

    print(f"  Selected hyperparameters: {selected_hyperparams}")
    print(f"  Validation errors for selected model: {selected_model_validation_errors} out of {X_val.shape[0] if X_val.shape[0] > 0 else 'N/A'}.")
    
    return selected_hyperparams, selected_model_validation_errors

##########################################################################
# --- Main Experiment Function ---
##########################################################################
def run_experiment(dataset_name, algorithm_names):
    print(f"\n{'='*80}")
    print(f"STARTING EXPERIMENT RUN")
    print(f"Dataset: {dataset_name}, Algorithms: {', '.join(algorithm_names)}")
    print(f"{'='*80}")

    try:
        X_full, y_full_binary = load_dataset(dataset_name)
    except Exception as e:
        print(f"CRITICAL Error: Failed to load dataset '{dataset_name}'. Aborting. Details: {type(e).__name__} {e}")
        return None

    rng_main_split = np.random.default_rng(MAIN_SEED)
    n_total = X_full.shape[0]
    perm_main_split = rng_main_split.permutation(n_total)
    split_idx_main = int(0.8 * n_total) 
    train_indices_main, test_indices_main = perm_main_split[:split_idx_main], perm_main_split[split_idx_main:]
    X_train_overall, y_train_overall_binary = X_full[train_indices_main], y_full_binary[train_indices_main]
    X_test, y_test_binary = X_full[test_indices_main], y_full_binary[test_indices_main]

    print(f"Overall data split: Training/Tuning data size: {X_train_overall.shape[0]}, Test data size: {X_test.shape[0]}")

    if X_train_overall.shape[0] == 0:
        print("CRITICAL Error: Training data size is zero after split. Aborting.")
        return None
    delta = 1.0 / (X_train_overall.shape[0]**2) 
    print(f"Delta (δ) for DP mechanisms = {delta:.2e}")

    hyperparam_master_config = get_hyperparameter_configs(dataset_name, X_train_overall.shape[0])
    experiment_results_all_algs = {}
    results_dir = f"./results_tuning_{dataset_name}_seed{MAIN_SEED}" # Include seed in results dir name
    create_directory(results_dir)

    for alg_name in algorithm_names:
        if alg_name not in hyperparam_master_config:
            print(f"Algorithm '{alg_name}' not in hyperparameter configurations. Skipping.")
            continue
        print(f"\n{'-'*60}\nProcessing Algorithm: {alg_name}\n{'-'*60}")
        alg_config = hyperparam_master_config[alg_name]
        alg_class, param_grid_for_alg = alg_config['class'], alg_config['params']
        alg_results_all_epsilons = {}

        for epsilon_total_experiment in EPSILON_VALUES:
            print(f"\nTesting with Total Epsilon = {epsilon_total_experiment}")
            current_epsilon_repeat_accuracies = []

            for repeat_num in range(NUM_REPEATS):
                current_seed_for_repeat = (MAIN_SEED + repeat_num + 
                                           EPSILON_VALUES.index(epsilon_total_experiment) * NUM_REPEATS + 
                                           abs(hash(alg_name)) % 10000)
                print(f"  Repeat {repeat_num + 1}/{NUM_REPEATS} (Repeat Seed Base: {current_seed_for_repeat})")

                epsilon_for_tuning_phase = epsilon_total_experiment / 2.0
                epsilon_for_final_model = epsilon_total_experiment / 2.0
                print(f"    Total Epsilon split: Tuning Phase Epsilon = {epsilon_for_tuning_phase:.3f}, Final Model Training Epsilon = {epsilon_for_final_model:.3f}")

                rng_repeat_data_split = np.random.default_rng(current_seed_for_repeat)
                n_train_overall_samples = X_train_overall.shape[0]
                perm_repeat_data_split = rng_repeat_data_split.permutation(n_train_overall_samples)
                split_idx_tuning_vs_final = int(0.75 * n_train_overall_samples) 
                tuning_data_indices = perm_repeat_data_split[:split_idx_tuning_vs_final]
                final_model_train_indices = perm_repeat_data_split[split_idx_tuning_vs_final:]

                X_tuning_data_current_repeat = X_train_overall[tuning_data_indices]
                y_tuning_data_binary_current_repeat = y_train_overall_binary[tuning_data_indices]
                X_final_model_train_data_current_repeat = X_train_overall[final_model_train_indices]
                y_final_model_train_data_binary_current_repeat = y_train_overall_binary[final_model_train_indices]

                min_required_tuning_data = MIN_SUBSET_SIZE * 2 # Need at least for 1 cand + 1 val for the check in private_hyperparameter_tuning
                if X_tuning_data_current_repeat.shape[0] < min_required_tuning_data or \
                   X_final_model_train_data_current_repeat.shape[0] < MIN_SUBSET_SIZE:
                    print(f"    Insufficient data for repeat {repeat_num+1} after splitting X_train_overall.")
                    print(f"      Tuning data size: {X_tuning_data_current_repeat.shape[0]} (need >={min_required_tuning_data}), "
                          f"Final model train data size: {X_final_model_train_data_current_repeat.shape[0]} (need >={MIN_SUBSET_SIZE}). Skipping repeat.")
                    current_epsilon_repeat_accuracies.append(0.0)
                    continue
                print(f"    Data for this repeat: Tuning Data size {X_tuning_data_current_repeat.shape[0]}, Final Model Train Data size {X_final_model_train_data_current_repeat.shape[0]}")

                selected_hyperparams, _ = private_hyperparameter_tuning(
                    alg_class, param_grid_for_alg,
                    X_tuning_data_current_repeat, y_tuning_data_binary_current_repeat,
                    epsilon_for_tuning_phase, delta,
                    random_state_base=current_seed_for_repeat + 1000
                )

                print(f"  Training final model with selected hyperparameters: {selected_hyperparams}")
                final_model_theta = None
                if selected_hyperparams and isinstance(selected_hyperparams, dict) and selected_hyperparams: # Check if tuning returned valid params
                    try:
                        # final_algorithm_instance = alg_class(verbose=False) # if alg takes verbose
                        final_algorithm_instance = alg_class()
                        final_model_theta, _, _ = final_algorithm_instance.run_classification(
                            x=X_final_model_train_data_current_repeat, 
                            y=y_final_model_train_data_binary_current_repeat,
                            epsilon=epsilon_for_final_model, delta=delta,
                            random_state=current_seed_for_repeat + 2000,
                            **selected_hyperparams
                        )
                        if final_model_theta is None or np.allclose(final_model_theta, 0.0, atol=1e-12):
                            print(f"    Final model training resulted in trivial or None theta.")
                            final_model_theta = None 
                    except Exception as e:
                        print(f"    Error during final model training: {type(e).__name__} {e}")
                        final_model_theta = None
                else:
                    print(f"    No valid hyperparameters from tuning. Skipping final model training.")

                test_accuracy = evaluate_model_accuracy(final_model_theta, X_test, y_test_binary)
                current_epsilon_repeat_accuracies.append(test_accuracy)
                print(f"    Test accuracy for repeat {repeat_num + 1} (Total Epsilon={epsilon_total_experiment}): {test_accuracy:.4f}")

            if current_epsilon_repeat_accuracies:
                mean_acc, std_acc = np.mean(current_epsilon_repeat_accuracies), np.std(current_epsilon_repeat_accuracies)
            else:
                mean_acc, std_acc = 0.0, 0.0
            alg_results_all_epsilons[epsilon_total_experiment] = {
                'mean': mean_acc, 'std': std_acc, 'individual': current_epsilon_repeat_accuracies}
            print(f"    Summary for Total Epsilon={epsilon_total_experiment}: Mean Accuracy = {mean_acc:.4f} +/- {std_acc:.4f}")
        experiment_results_all_algs[alg_name] = alg_results_all_epsilons
    
    save_results(experiment_results_all_algs, dataset_name, results_dir)
    plot_results(experiment_results_all_algs, dataset_name, results_dir, X_full, y_full_binary)
    return experiment_results_all_algs

def save_results(results_all_algs, dataset_name, results_dir):
    csv_filename = os.path.join(results_dir, f"{dataset_name}_tuning_accuracies_summary.csv")
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['Algorithm', 'Epsilon_Total'] + \
                     [f'Repeat_{i+1}_Acc' for i in range(NUM_REPEATS)] + \
                     ['Mean_Acc', 'Std_Acc']
            writer.writerow(header)
            for alg_name, alg_epsilon_results in results_all_algs.items():
                for epsilon_total, eps_data in alg_epsilon_results.items():
                    individual_accuracies = eps_data.get('individual', [])
                    padded_accuracies = (list(individual_accuracies) + [0.0] * NUM_REPEATS)[:NUM_REPEATS]
                    row = ([alg_name, epsilon_total] + 
                           [f"{acc:.4f}" for acc in padded_accuracies] + 
                           [f"{eps_data.get('mean', 0.0):.4f}", f"{eps_data.get('std', 0.0):.4f}"])
                    writer.writerow(row)
        print(f"\nAggregated accuracy results saved to: {csv_filename}")
    except IOError as e:
        print(f"Error saving results to CSV: {type(e).__name__} {e}")

def plot_results(results_all_algs, dataset_name, results_dir, X_full_for_baseline, y_full_binary_for_baseline):
    plt.figure(figsize=(12, 8))
    num_algorithms = len(results_all_algs)
    colors = plt.cm.viridis(np.linspace(0, 0.9, num_algorithms if num_algorithms > 0 else 1)) 
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
    
    for i, (alg_name, alg_epsilon_results) in enumerate(results_all_algs.items()):
        epsilons_plot = sorted(alg_epsilon_results.keys())
        means_plot = [alg_epsilon_results[eps]['mean'] for eps in epsilons_plot]
        stds_plot = [alg_epsilon_results[eps]['std'] for eps in epsilons_plot]
        plt.errorbar(epsilons_plot, means_plot, yerr=stds_plot, 
                     label=alg_name, color=colors[i % len(colors)], 
                     marker=markers[i % len(markers)], 
                     capsize=5, elinewidth=1.5, linewidth=2, markersize=7, markerfacecolor='white', markeredgewidth=1.5)
    try:
        rng_baseline_split = np.random.default_rng(MAIN_SEED)
        n_total_baseline = X_full_for_baseline.shape[0]
        perm_baseline_split = rng_baseline_split.permutation(n_total_baseline)
        split_idx_baseline = int(0.8 * n_total_baseline)
        train_indices_baseline, test_indices_baseline = perm_baseline_split[:split_idx_baseline], perm_baseline_split[split_idx_baseline:]
        X_train_baseline, y_train_baseline_sklearn = X_full_for_baseline[train_indices_baseline], np.where(y_full_binary_for_baseline[train_indices_baseline] == 1, 1, 0) 
        X_test_baseline, y_test_baseline_sklearn = X_full_for_baseline[test_indices_baseline], np.where(y_full_binary_for_baseline[test_indices_baseline] == 1, 1, 0)

        if X_train_baseline.shape[0] > 0 and X_test_baseline.shape[0] > 0:
            baseline_model = sklearn.linear_model.LogisticRegression(
                solver='liblinear', C=1.0, fit_intercept=True, 
                random_state=MAIN_SEED, max_iter=2000)
            baseline_model.fit(X_train_baseline, y_train_baseline_sklearn)
            baseline_accuracy = baseline_model.score(X_test_baseline, y_test_baseline_sklearn)
            plt.axhline(y=baseline_accuracy, color='dimgrey', linestyle=':', 
                        linewidth=2.5, label=f'Non-Private LR Baseline ({baseline_accuracy:.3f})')
            print(f"Non-private Logistic Regression baseline accuracy: {baseline_accuracy:.4f}")
        else:
            print("Skipping non-private baseline: not enough data after split.")
    except Exception as e:
        print(f"Could not compute or plot non-private baseline: {type(e).__name__} {e}")

    plt.xscale('log')
    plt.xlabel('Total Privacy Budget (Epsilon_total)', fontsize=14)
    plt.ylabel('Test Accuracy', fontsize=14)
    plt.title(f'Accuracy vs. Privacy Budget - {dataset_name.title()}', fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=10, frameon=True, shadow=True)
    plt.grid(True, which="both", linestyle='--', linewidth=0.7, alpha=0.7)
    current_ymin, current_ymax = plt.gca().get_ylim()
    plot_ymin = max(0, current_ymin - 0.05) if not math.isnan(current_ymin) else 0
    plot_ymax = min(1.0, current_ymax + 0.05 if current_ymax < 0.95 and not math.isnan(current_ymax) else 1.0)
    plt.ylim(plot_ymin, plot_ymax)
    plt.xticks(EPSILON_VALUES, labels=[str(ep) for ep in EPSILON_VALUES]) 
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()

    plot_filename = os.path.join(results_dir, f"{dataset_name}_accuracy_vs_epsilon_plot.png")
    try:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_filename}")
    except IOError as e:
        print(f"Error saving plot: {type(e).__name__} {e}")
    plt.close()

def main():
    # if sys.platform.startswith('win'):
    #     import multiprocessing
    #     multiprocessing.freeze_support()

    if len(sys.argv) < 3:
        print(f"Usage: python {os.path.basename(sys.argv[0])} <dataset_name> <algorithm_choice | ALL>")
        print(f"Available datasets: {', '.join(AVAILABLE_DATASETS)}")
        print(f"Algorithm choices: DPSGD, AMP, TukeyEM, ALL")
        sys.exit(1)
    
    dataset_name_arg = sys.argv[1].lower()
    algorithm_choice_arg = sys.argv[2].upper()

    if dataset_name_arg not in AVAILABLE_DATASETS:
        print(f"Unknown dataset: '{dataset_name_arg}'. Choose from: {AVAILABLE_DATASETS}")
        sys.exit(1)
    
    valid_algorithms_input = ['DPSGD', 'AMP', 'TUKEYEM'] 
    algorithms_to_run = []
    if algorithm_choice_arg == 'ALL':
        algorithms_to_run = ['DPSGD', 'AMP', 'TukeyEM']
    elif algorithm_choice_arg in valid_algorithms_input:
        internal_alg_name = 'TukeyEM' if algorithm_choice_arg == 'TUKEYEM' else algorithm_choice_arg
        algorithms_to_run = [internal_alg_name]
    else:
        print(f"Unknown algorithm choice: '{algorithm_choice_arg}'. Choose from: {valid_algorithms_input} or ALL.")
        sys.exit(1)

    print(f"CONFIGURATION:")
    print(f"  Dataset: '{dataset_name_arg}'")
    print(f"  Algorithm(s): {', '.join(algorithms_to_run)}")
    print(f"  Epsilon values: {EPSILON_VALUES}")
    print(f"  Repeats per epsilon: {NUM_REPEATS}")
    print(f"  Parallel Cores: {CORES}")
    print(f"  Main Seed: {MAIN_SEED}")

    experiment_start_time = process_time()
    results = run_experiment(dataset_name_arg, algorithms_to_run)
    experiment_end_time = process_time()

    if results:
        print(f"\n{'='*80}")
        print(f"EXPERIMENT COMPLETED SUCCESSFULLY!")
        total_time_seconds = experiment_end_time - experiment_start_time
        print(f"Total execution time: {total_time_seconds:.2f} seconds ({total_time_seconds/60:.2f} minutes)")
        print(f"{'='*80}\nSUMMARY OF MEAN ACCURACIES:")
        for alg_name_summary, alg_results_summary in results.items():
            print(f"\nAlgorithm: {alg_name_summary}")
            for epsilon_total_summary, data_summary in sorted(alg_results_summary.items()):
                individual_scores_str = [f'{x:.3f}' for x in data_summary.get('individual', [])]
                print(f"  Epsilon_total={epsilon_total_summary:<4.1f}: Mean Acc = {data_summary['mean']:.4f} +/- {data_summary['std']:.4f} "
                      f"(Indiv: [{', '.join(individual_scores_str)}])")
    else:
        print(f"\nEXPERIMENT FAILED OR PRODUCED NO RESULTS.")

if __name__ == '__main__':
    main()