import os
import sys
import csv
import math
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
from time import process_time
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from multiprocessing import Pool
from itertools import product
from common.common import Algorithm, compute_classification_counts, compute_multiclass_counts
from common.clipping import clip_rows # You mentioned this prints clipping stats
from dpsgd_lr_real import DPSGDLR # Your refined DPSGD algorithm
from AMP_lr_real import ApproximateMinimaPerturbationLR # Your refined AMP algorithm

##########################################################################
# --- Global Constants and Configuration ---
dataset_location = './datasets/data'
multivariate_datasets = ['covertype', 'mnist', 'o185', 'o313', 'o4550', 'PEMS', 'wine']
sparse_datasets = ['farm', 'dexter', 'dorothea', 'realsim', 'rcv1', 'news20']
data2shape = {
    'farm': (4143, 54877), 'dexter': (300, 20000), 'dorothea': (800, 100000),
    'realsim': (72309, 20958), 'rcv1': (50000, 47236), 'news20': (8870, 117049)
}

NUM_REPEATS = 2
CORES = 4
all_eps_list = [0.1, 0.5, 1.0, 5.0, 10.0]
MAIN_SEED = 42 # Master seed for all repetitions
##########################################################################

##########################################################################
# --- Helper Functions ---
##########################################################################
def build_binary_ys_for_class(y_one_hot_or_flat, class_index):
    """Converts labels to binary {-1, 1} for a specific class in OvR."""
    if y_one_hot_or_flat.ndim == 1:
        unique_labels = np.unique(y_one_hot_or_flat)
        if len(unique_labels) <= 2: # Binary or effectively binary
            positive_label_val = unique_labels[-1] # Assume last unique label is positive if more than one
            if len(unique_labels) == 1: # Only one label value present
                positive_label_val = unique_labels[0] # Treat this single label as positive for OvR context
            return np.where(y_one_hot_or_flat == positive_label_val, 1, -1)
        else: # Multiclass flat labels (0,1,2...)
            return np.where(y_one_hot_or_flat == class_index, 1, -1)
    elif y_one_hot_or_flat.ndim == 2: # One-hot encoded
        if y_one_hot_or_flat.shape[1] <= class_index:
            raise ValueError(f"class_index {class_index} out of bounds for y_one_hot with shape {y_one_hot_or_flat.shape}")
        return np.where(y_one_hot_or_flat[:, class_index] == 1, 1, -1)
    else:
        raise ValueError("Unsupported y shape for build_binary_ys_for_class")

def dict_product(dicts):
    """Generates Cartesian product of dictionaries."""
    return (dict(zip(dicts, x)) for x in product(*dicts.values()))

def progress_bar(pct):
    """Displays a simple progress bar in the console."""
    i = int(pct)
    sys.stdout.write('\r')
    sys.stdout.write("\033[K[ %-20s ] %d%%" % ('=' * int(i / 5), i))
    sys.stdout.flush()

def create_directory(directory_name):
    """Creates a directory if it doesn't already exist."""
    try:
        os.makedirs(directory_name, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {directory_name}: {e}")

##########################################################################
# --- Task Functions for Parallel Execution (Must be Top-Level) ---
##########################################################################
def train_candidate_model_task(args_dict):
    """Trains a single candidate model for Algorithm 3."""
    alg_class = args_dict['alg_class']
    train_x_clipped = args_dict['train_x_clipped'] # Expect pre-clipped data for this task
    train_y_orig_subset = args_dict['train_y_orig_subset']
    epsilon_budget_for_train = args_dict['epsilon_budget_for_train']
    delta_privacy = args_dict['delta_privacy']
    hyper_params = args_dict['hyper_params']
    is_multi = args_dict['is_multi']
    num_classes_total = args_dict['num_classes_total']
    k_class_ovr = args_dict['k_class_ovr']
    current_task_seed = args_dict['current_task_seed']
    
    run_args = {'x': train_x_clipped, 'delta': delta_privacy,
                'random_state': current_task_seed, **hyper_params}

    # Instantiate algorithm class to call instance methods like name() and run_classification()
    alg_instance = alg_class()
    current_alg_name = alg_instance.name # Access name via instance

    if is_multi and num_classes_total > 0:
        run_args['y'] = build_binary_ys_for_class(train_y_orig_subset, k_class_ovr)
        # Epsilon splitting for OvR models (DPSGD and AMP)
        if current_alg_name in ["DPSGD-LR", "AMP-LR"]:
             run_args['epsilon'] = epsilon_budget_for_train / num_classes_total
        else: # Fallback for other potential algorithms
             run_args['epsilon'] = epsilon_budget_for_train
    else: # Binary case
        run_args['epsilon'] = epsilon_budget_for_train
        run_args['y'] = build_binary_ys_for_class(train_y_orig_subset, 0)
    
    try:
        model_theta, l_val, gamma_val = alg_instance.run_classification(**run_args)
        return {'theta': model_theta, 'L': l_val, 'gamma': gamma_val, 
                'hyper_params': hyper_params, 'k_class_ovr': k_class_ovr if is_multi else None}
    except Exception as e:
        print(f"\nError training candidate model (seed {current_task_seed}): {e}")
        return {'theta': None, 'L': hyper_params.get('L',1.0), 'gamma': np.nan, 
                'hyper_params': hyper_params, 'k_class_ovr': k_class_ovr if is_multi else None}

def evaluate_candidate_on_sval_task(args_dict):
    """Evaluates a candidate model on S_val to get error count (chi_j)."""
    model_theta_obj = args_dict['model_theta_obj']
    s_val_features_unclipped = args_dict['s_val_features_unclipped']
    s_val_labels_orig = args_dict['s_val_labels_orig']
    is_multi = args_dict['is_multi']
    L_val_model_clipping = args_dict['L_val_model_clipping']

    if model_theta_obj is None: # Model training failed
        return s_val_features_unclipped.shape[0] if s_val_features_unclipped is not None and s_val_features_unclipped.shape[0] > 0 else 1

    # Clip S_val features with the L specific to this model
    s_val_features_eval_clipped = clip_rows(s_val_features_unclipped, 2, L_val_model_clipping)

    if is_multi:
        y_true_sval_eval = np.argmax(s_val_labels_orig, axis=1) if s_val_labels_orig.ndim > 1 else s_val_labels_orig.astype(int)
        correct, incorrect = compute_multiclass_counts(s_val_features_eval_clipped, y_true_sval_eval, model_theta_obj)
    else:
        y_true_sval_binary_eval = build_binary_ys_for_class(s_val_labels_orig, 0)
        correct, incorrect = compute_classification_counts(s_val_features_eval_clipped, y_true_sval_binary_eval, model_theta_obj)
    return incorrect # Return number of errors (chi_j)

##########################################################################
# --- Main Tuning Function (Modified for Algorithm 3) ---
##########################################################################
def main():
    print("Starting DP Algorithm Tuning Script (Implementing Algorithm 3 from Wu et al.)...")
    np.seterr(over='ignore', divide='ignore', invalid='ignore')

    # --- Argument Parsing (Simplified) ---
    if len(sys.argv) != 3:
        print('Usage: python ttuning.py <dataset_name> <algorithm_choice>')
        print("Algorithm choices: DPSGD, AMP, ALL")
        sys.exit(1)

    dataset_name = sys.argv[1]
    algorithm_choice_arg = sys.argv[2].upper()
    model_name_hint = 'LR' # Fixed, as DP algorithms are Logistic Regression

    if algorithm_choice_arg == 'DPSGD': alg_name_list = ['DPSGD']
    elif algorithm_choice_arg == 'AMP': alg_name_list = ['AMP']
    elif algorithm_choice_arg == 'ALL': alg_name_list = ['DPSGD', 'AMP']
    else: print(f"Error: Unknown algorithm choice '{sys.argv[2]}'. Use DPSGD, AMP, or ALL."); sys.exit(1)

    print(f"--- Tuning algorithm(s) using Algorithm 3: {alg_name_list} ---")
    print(f"Dataset: {dataset_name}, Model Type for DP Algos: {model_name_hint}")

    results_base = "./results_alg3"; results_rough_model = os.path.join(results_base, "rough_results", model_name_hint)
    results_graphs_model = os.path.join(results_base, "graphs", model_name_hint)
    for path_dir in [results_base, os.path.join(results_base, "rough_results"), results_rough_model, os.path.join(results_base, "graphs"), results_graphs_model]: create_directory(path_dir)

    # --- Dataset Loading ---
    try:
        if dataset_name in data2shape and dataset_name in sparse_datasets:
            print(f"Loading sparse dataset: {dataset_name}"); d_path = os.path.join(dataset_location, f'{dataset_name}_processed_d.npy'); idx_path = os.path.join(dataset_location, f'{dataset_name}_processed_indices.npy'); ptr_path = os.path.join(dataset_location, f'{dataset_name}_processed_indptr.npy'); y_path = os.path.join(dataset_location, f'{dataset_name}_processed_y.npy')
            if not all(map(os.path.exists, [d_path, idx_path, ptr_path, y_path])): raise FileNotFoundError(f"Sparse data files missing for {dataset_name}")
            data_vals = np.load(d_path, allow_pickle=True); indices = np.load(idx_path, allow_pickle=True); indptr = np.load(ptr_path, allow_pickle=True); features = csr_matrix((data_vals, indices, indptr), shape=data2shape[dataset_name]); labels_orig = np.load(y_path, allow_pickle=True).astype(float)
        elif os.path.exists(os.path.join(dataset_location, f'{dataset_name}_processed_x.npy')):
            print(f"Loading dense dataset: {dataset_name}"); x_path = os.path.join(dataset_location, f'{dataset_name}_processed_x.npy'); y_path = os.path.join(dataset_location, f'{dataset_name}_processed_y.npy')
            if not all(map(os.path.exists, [x_path, y_path])): raise FileNotFoundError(f"Dense data files missing for {dataset_name}")
            features = np.load(x_path, allow_pickle=True).astype(float); labels_orig = np.load(y_path, allow_pickle=True).astype(float)
        else: raise FileNotFoundError(f"No processed data files found for '{dataset_name}'")
    except Exception as e: print(f"Error loading dataset '{dataset_name}': {e}. Exiting."); sys.exit(1)
    print(f"Dataset loaded: Features={features.shape}, Labels={labels_orig.shape}")

    # --- Label and Data Prep ---
    is_multivariate = dataset_name in multivariate_datasets or \
                      (labels_orig.ndim > 1 and labels_orig.shape[1] > 1) or \
                      (labels_orig.ndim == 1 and len(np.unique(labels_orig)) > 2)
    effective_num_classes_total = 1
    if is_multivariate:
        if labels_orig.ndim == 1: num_classes_val = len(np.unique(labels_orig))
        else: num_classes_val = labels_orig.shape[1]
        effective_num_classes_total = num_classes_val
        print(f"Detected {num_classes_val} classes (multivariate).")
    else: print("Detected binary classification.")

    # --- Data Splitting (Main Train/Test for final evaluation) ---
    n_total = features.shape[0]
    if n_total <= 1: print("Error: Not enough data samples."); sys.exit(1)
    main_test_split_idx = int(n_total * 0.8)
    original_full_train_features_unclipped = features[:main_test_split_idx]
    public_test_features_unclipped = features[main_test_split_idx:]
    original_full_train_labels = labels_orig[:main_test_split_idx] # Used for S_j and S_val labels
    public_test_labels = labels_orig[main_test_split_idx:]       # Used for final evaluation of selected model
    n_train_full_orig = original_full_train_features_unclipped.shape[0]

    if n_train_full_orig == 0 or public_test_features_unclipped.shape[0] == 0: print("Error: Original train or public test set empty."); sys.exit(1)
    print(f"Original Full Train size: {n_train_full_orig}, Public Test size: {public_test_features_unclipped.shape[0]}")

    # --- Non-Private Baseline ---
    baseline_accuracy = 0.0
    labels_sklearn_baseline = labels_orig.astype(int) # Prepare labels for sklearn
    if is_multivariate and labels_orig.ndim > 1: labels_sklearn_baseline = np.argmax(labels_orig, axis=1)
    elif not is_multivariate : # Ensure binary labels are 0/1 for scikit-learn
        unique_raw_labels_sklearn = np.unique(labels_sklearn_baseline)
        if not np.all(np.isin(unique_raw_labels_sklearn, [0, 1])):
            positive_label_val_sklearn = np.max(unique_raw_labels_sklearn) if len(unique_raw_labels_sklearn) > 0 else 1
            labels_sklearn_baseline = np.where(labels_sklearn_baseline == positive_label_val_sklearn, 1, 0)
    
    train_labels_sklearn_baseline = labels_sklearn_baseline[:main_test_split_idx]
    test_labels_sklearn_baseline = labels_sklearn_baseline[main_test_split_idx:]
    try:
        baseline_model = SklearnLogisticRegression(max_iter=2000, solver='liblinear', penalty='l2', C=1.0, multi_class='ovr' if is_multivariate and effective_num_classes_total > 1 else 'auto', random_state=MAIN_SEED)
        baseline_model.fit(original_full_train_features_unclipped, train_labels_sklearn_baseline)
        predicted_labels_baseline = baseline_model.predict(public_test_features_unclipped)
        baseline_accuracy = np.mean(predicted_labels_baseline == test_labels_sklearn_baseline)
        print(f"Baseline sklearn LR accuracy: {baseline_accuracy:.4f}")
    except Exception as e: print(f"Scikit-learn baseline failed: {e}")

    # --- Epsilon List (Fixed) and Output Files ---
    eps_to_run = sorted(list(set(e for e in all_eps_list if e > 0)))
    if not eps_to_run: print("Error: all_eps_list is invalid or empty. Exiting."); sys.exit(1)
    num_eps_values = len(eps_to_run); print(f"Epsilon values for Algorithm 3: {eps_to_run}")
    
    eps_suffix = '_defaultEpsFixed'
    output_basename = f"Alg3_{'_'.join(alg_name_list)}_{dataset_name}_{model_name_hint}{eps_suffix}"
    accfile_path = os.path.join(results_rough_model, output_basename + '.acc'); stdfile_path = os.path.join(results_rough_model, output_basename + '.std')
    logfile_path = os.path.join(results_rough_model, output_basename + '.log'); plot_filename = os.path.join(results_graphs_model, output_basename + '_acc_vs_eps.png')

    # --- Algorithm Config ---
    delta_privacy = 1 / (n_train_full_orig**2) if n_train_full_orig > 1 else 1e-3
    print(f"Using delta = {delta_privacy:.2e} for all DP algorithm trainings")
    hyper_configs = { # Ensure hyperparameter keys match expected arguments of your algorithm classes
        'DPSGD': {'class': DPSGDLR, 'hyper': {'lambda_param': [1e-3, 1e-2], 'num_epochs': [10, 50], 'batch_size': [50], 'L': [1.0]}},
        'AMP': {'class': ApproximateMinimaPerturbationLR, 'hyper': {
            'l2_constraint': [1.0, 10.0], 'eps_frac_obj_noise_calc': [0.9, 0.99], 
            'eps_frac_output_noise': [0.01, 0.1], 'gamma': [(1.0 / n_train_full_orig if n_train_full_orig > 0 else 1.0)], 'L': [1.0, 10.0]}}
    }

    acc_matrix = np.zeros([1 + len(alg_name_list), num_eps_values]); std_matrix = np.zeros([1 + len(alg_name_list), num_eps_values])
    acc_matrix[0, :] = baseline_accuracy; std_matrix[0, :] = 0.0

    # --- Main Experiment Loop for Algorithm 3 ---
    with open(accfile_path, 'w') as accfile, open(stdfile_path, 'w') as stdfile, open(logfile_path, 'w') as logfile:
        csv_header = "Algorithm," + ",".join([f"eps={e:.4f}" for e in eps_to_run])
        print(csv_header, file=accfile); print(csv_header, file=stdfile)
        print(f"NonPrivate,{','.join(map(lambda x: f'{x:.4f}', acc_matrix[0]))}", file=accfile)
        print(f"NonPrivate,{','.join(map(lambda x: f'{x:.4f}', std_matrix[0]))}", file=stdfile); accfile.flush(); stdfile.flush()

        for alg_idx, alg_name_current in enumerate(alg_name_list):
            print(f'\n--- Applying Algorithm 3 for: {alg_name_current} ---')
            config = hyper_configs[alg_name_current]; AlgClass = config['class']
            hyper_param_sets = list(dict_product(config['hyper'])); l_num_hyper_configs = len(hyper_param_sets)
            if l_num_hyper_configs == 0: print(f"No hyperparams for {alg_name_current}, skipping."); continue
            
            # Adjusted log_keys for Algorithm 3 context
            log_keys_alg3 = ['eps', 'repeat_num', 'selected_hyper_idx', 'selected_hyper_params_str', 
                             'num_errors_on_sval_for_selected', 'selected_model_acc_on_public_test', 
                             'selected_model_L', 'selected_model_gamma']
            print(f'\n--- Detailed Log for Algorithm 3 ({alg_name_current}) ---', file=logfile)
            print('\t'.join(log_keys_alg3), file=logfile); logfile.flush()

            accuracies_for_plot_alg = np.full(num_eps_values, np.nan)
            stds_for_plot_alg = np.full(num_eps_values, np.nan)

            for i_eps, current_epsilon_total_for_alg3 in enumerate(eps_to_run):
                print(f"\n  Epsilon = {current_epsilon_total_for_alg3:.4f} for {alg_name_current} using Algorithm 3")
                final_selected_model_accuracies_one_eps = []

                for i_repeat in range(NUM_REPEATS):
                    print(f"    Repeat {i_repeat + 1}/{NUM_REPEATS}")
                    
                    current_repeat_base_seed_offset = alg_idx * 1000000 + i_eps * 100000 + i_repeat * 10000
                    rng_data_splitter = np.random.default_rng(MAIN_SEED + current_repeat_base_seed_offset)
                    perm_indices_train = rng_data_splitter.permutation(n_train_full_orig)
                    
                    min_data_for_split = l_num_hyper_configs + 1
                    if n_train_full_orig < min_data_for_split:
                        print(f"    Warning: Not enough training data ({n_train_full_orig}) for {min_data_for_split} splits. Skipping."); break
                    split_size = n_train_full_orig // min_data_for_split
                    if split_size == 0: print(f"    Split size 0 (n_train_full_orig={n_train_full_orig}, l_configs={l_num_hyper_configs}). Skipping."); break

                    s_val_indices = perm_indices_train[l_num_hyper_configs * split_size:] # Remainder becomes S_val
                    s_val_features_unclipped_current = original_full_train_features_unclipped[s_val_indices]
                    s_val_labels_current = original_full_train_labels[s_val_indices]
                    if s_val_features_unclipped_current.shape[0] == 0: print(f"    S_val empty. Skipping repeat."); continue

                    candidate_training_specs = []
                    for j_hyper, h_params_j in enumerate(hyper_param_sets):
                        s_j_indices = perm_indices_train[j_hyper * split_size:(j_hyper + 1) * split_size]
                        if len(s_j_indices) == 0: candidate_training_specs.append(None); continue

                        L_j_train = h_params_j.get('L', 1.0)
                        s_j_features_clipped = clip_rows(original_full_train_features_unclipped[s_j_indices], 2, L_j_train)
                        s_j_labels_subset_train = original_full_train_labels[s_j_indices]
                        
                        base_task_seed_offset_hyper = current_repeat_base_seed_offset + j_hyper * (effective_num_classes_total +1) # Unique per hyper
                        
                        if is_multivariate and effective_num_classes_total > 1:
                            for k_ovr in range(effective_num_classes_total):
                                candidate_training_specs.append({
                                    'alg_class': AlgClass, 'train_x_clipped': s_j_features_clipped, 'train_y_orig_subset': s_j_labels_subset_train,
                                    'epsilon_budget_for_train': current_epsilon_total_for_alg3, 'delta_privacy': delta_privacy,
                                    'hyper_params': h_params_j, 'is_multi': True, 'num_classes_total': effective_num_classes_total,
                                    'k_class_ovr': k_ovr, 'current_task_seed': MAIN_SEED + base_task_seed_offset_hyper + k_ovr})
                        else:
                            candidate_training_specs.append({
                                'alg_class': AlgClass, 'train_x_clipped': s_j_features_clipped, 'train_y_orig_subset': s_j_labels_subset_train,
                                'epsilon_budget_for_train': current_epsilon_total_for_alg3, 'delta_privacy': delta_privacy,
                                'hyper_params': h_params_j, 'is_multi': False, 'num_classes_total': 1,
                                'k_class_ovr': 0, 'current_task_seed': MAIN_SEED + base_task_seed_offset_hyper})
                    
                    trained_model_parts_results = [] # Flat list of results from train_candidate_model_task
                    if candidate_training_specs:
                        with Pool(CORES) as pool:
                            async_results_train = [pool.apply_async(train_candidate_model_task, (spec,)) for spec in candidate_training_specs if spec is not None]
                            for i_task_train, res_async_train in enumerate(async_results_train):
                                progress_bar((i_task_train + 1) * 100 / len(async_results_train))
                                try: trained_model_parts_results.append(res_async_train.get(timeout=3600))
                                except Exception as e: print(f"\nError/Timeout in parallel candidate training: {e}"); trained_model_parts_results.append(None)
                        print("\n    Candidate model training collection finished.")
                    
                    # --- Aggregate OvR models ---
                    aggregated_candidate_models_info = [] # List of full model info for each of l_num_hyper_configs
                    part_idx_counter = 0
                    for j_hyper_agg in range(l_num_hyper_configs):
                        parts_for_this_hyper_set = []
                        failed_hyper_set = False
                        for _ in range(effective_num_classes_total):
                            if part_idx_counter < len(trained_model_parts_results):
                                part_res = trained_model_parts_results[part_idx_counter]
                                if part_res is None or part_res['theta'] is None: failed_hyper_set = True
                                parts_for_this_hyper_set.append(part_res)
                                part_idx_counter += 1
                            else: failed_hyper_set = True; break # Not enough parts
                        
                        if failed_hyper_set or not parts_for_this_hyper_set : # Or if hyper_spec was None
                             aggregated_candidate_models_info.append(None) # Mark this hyper_config as failed
                             continue

                        h_params_agg = parts_for_this_hyper_set[0]['hyper_params']
                        L_val_agg = parts_for_this_hyper_set[0]['L']
                        gammas_agg = [p['gamma'] for p in parts_for_this_hyper_set if p and p['gamma'] is not None and not np.isnan(p['gamma'])]
                        avg_gamma_agg = np.mean(gammas_agg) if gammas_agg else np.nan
                        
                        model_obj_agg = [p['theta'] for p in parts_for_this_hyper_set] if (is_multivariate and effective_num_classes_total > 1) else parts_for_this_hyper_set[0]['theta']
                        aggregated_candidate_models_info.append({'model_obj': model_obj_agg, 'hyper_params': h_params_agg, 'L_val': L_val_agg, 'gamma': avg_gamma_agg})

                    # --- Evaluate candidates on S_val to get error_counts_chi ---
                    error_counts_chi = []
                    valid_candidates_for_exp_mech = []
                    for model_info_for_sval in aggregated_candidate_models_info:
                        if model_info_for_sval is None or model_info_for_sval['model_obj'] is None:
                            error_counts_chi.append(s_val_features_unclipped_current.shape[0]) # Max error
                            continue
                        
                        sval_task_args = {'model_theta_obj': model_info_for_sval['model_obj'], 
                                          's_val_features_unclipped': s_val_features_unclipped_current, 
                                          's_val_labels_orig': s_val_labels_current, 
                                          'is_multi': is_multivariate and effective_num_classes_total > 1, 
                                          'L_val_model_clipping': model_info_for_sval['L_val']}
                        try:
                            errors_on_sval = evaluate_candidate_on_sval_task(sval_task_args)
                            error_counts_chi.append(errors_on_sval)
                            valid_candidates_for_exp_mech.append(model_info_for_sval)
                        except Exception as e:
                            print(f"\nError during S_val eval for hyper {model_info_for_sval['hyper_params']}: {e}")
                            error_counts_chi.append(s_val_features_unclipped_current.shape[0]) # Max error

                    if not valid_candidates_for_exp_mech: print(f"    Warning: No valid models for ExpMech after S_val eval. Skipping repeat."); final_selected_model_accuracies_one_eps.append(np.nan); continue
                    
                    # Use error_counts_chi corresponding to valid_candidates_for_exp_mech
                    # Need to map indices if some candidates failed and were not added to valid_candidates_for_exp_mech
                    # For simplicity, assume error_counts_chi was populated correctly for all, or filter it now.
                    # Let's assume error_counts_chi corresponds one-to-one with hyper_param_sets initially,
                    # and we select from valid_candidates_for_exp_mech using filtered error_counts.
                    
                    # Filter error_counts_chi for valid_candidates_for_exp_mech
                    # This is tricky if aggregated_candidate_models_info had Nones.
                    # Let's rebuild error_counts_chi for valid candidates only for simplicity:
                    error_counts_chi_valid = []
                    temp_valid_candidates = []
                    for j_hyper_final, h_params_final_j in enumerate(hyper_param_sets):
                        # Find if this hyper_param_set resulted in a valid model
                        found_valid = False
                        for valid_model_info_item in valid_candidates_for_exp_mech:
                            if valid_model_info_item['hyper_params'] == h_params_final_j: # Naive dict comparison
                                # Re-evaluate this specific valid model on S_val to get its chi
                                sval_task_args_final = {'model_theta_obj': valid_model_info_item['model_obj'], 
                                                        's_val_features_unclipped': s_val_features_unclipped_current, 
                                                        's_val_labels_orig': s_val_labels_current, 
                                                        'is_multi': is_multivariate and effective_num_classes_total > 1, 
                                                        'L_val_model_clipping': valid_model_info_item['L_val']}
                                error_counts_chi_valid.append(evaluate_candidate_on_sval_task(sval_task_args_final))
                                temp_valid_candidates.append(valid_model_info_item)
                                found_valid = True
                                break
                        if not found_valid: # This hyper_param set led to a failed model earlier
                             error_counts_chi_valid.append(s_val_features_unclipped_current.shape[0]) # Max error
                    
                    valid_candidates_for_exp_mech = temp_valid_candidates # Update to only those re-evaluated
                    if not valid_candidates_for_exp_mech: print(f"    Warning: Still no valid models for ExpMech. Skipping repeat."); final_selected_model_accuracies_one_eps.append(np.nan); continue


                    # --- Exponential Mechanism ---
                    scores_exp = np.array([-err for err in error_counts_chi_valid]) # Use errors of valid candidates
                    scores_shifted_exp = scores_exp - np.max(scores_exp) # Handles if all scores are bad
                    exp_terms_exp = np.exp(current_epsilon_total_for_alg3 * scores_shifted_exp / 2.0)
                    sum_exp_terms_exp = np.sum(exp_terms_exp)

                    if sum_exp_terms_exp <= 0 or np.isinf(sum_exp_terms_exp) or np.isnan(sum_exp_terms_exp) or len(valid_candidates_for_exp_mech) == 0:
                        print("    Warning: ExpMech probabilities ill-defined or no valid candidates. Selecting uniformly if possible, else skipping.")
                        if not valid_candidates_for_exp_mech: final_selected_model_accuracies_one_eps.append(np.nan); continue
                        probabilities_exp = np.ones(len(valid_candidates_for_exp_mech)) / len(valid_candidates_for_exp_mech)
                    else:
                        probabilities_exp = exp_terms_exp / sum_exp_terms_exp
                    probabilities_exp /= np.sum(probabilities_exp) # Ensure sum to 1

                    choice_rng_exp = np.random.default_rng(MAIN_SEED + current_repeat_base_seed_offset + l_num_hyper_configs + 1) # Seed for choice
                    try:
                        selected_idx_in_valid_list = choice_rng_exp.choice(len(valid_candidates_for_exp_mech), p=probabilities_exp)
                        selected_model_info_final = valid_candidates_for_exp_mech[selected_idx_in_valid_list]
                        # Find original index in hyper_param_sets for logging
                        original_selected_hyper_idx = -1
                        for idx, hps in enumerate(hyper_param_sets):
                            if hps == selected_model_info_final['hyper_params']: original_selected_hyper_idx = idx; break
                    except ValueError as e: # If probabilities don't sum to 1 or other choice error
                        print(f"    Warning: ExpMech choice error ({e}). Selecting uniformly.");
                        selected_idx_in_valid_list = choice_rng_exp.choice(len(valid_candidates_for_exp_mech))
                        selected_model_info_final = valid_candidates_for_exp_mech[selected_idx_in_valid_list]
                        original_selected_hyper_idx = -1 # Cannot easily determine original index

                    # --- Evaluate Selected Model on Public Test Set ---
                    L_final_eval = selected_model_info_final['L_val']
                    final_test_features_eval_clipped = clip_rows(public_test_features_unclipped, 2, L_final_eval) # Use unclipped public test
                    
                    final_y_true_public_test = public_test_labels
                    is_multi_final_eval = isinstance(selected_model_info_final['model_obj'], list) and len(selected_model_info_final['model_obj']) > 1
                    eval_func_public_test = compute_multiclass_counts if is_multi_final_eval else compute_classification_counts
                    if is_multi_final_eval:
                        final_y_true_public_test = np.argmax(public_test_labels, axis=1) if public_test_labels.ndim > 1 else public_test_labels.astype(int)
                    else:
                        final_y_true_public_test = build_binary_ys_for_class(public_test_labels, 0)

                    correct_final_pub, incorrect_final_pub = eval_func_public_test(final_test_features_eval_clipped, final_y_true_public_test, selected_model_info_final['model_obj'])
                    acc_final_selected_pub = correct_final_pub / (correct_final_pub + incorrect_final_pub) if (correct_final_pub + incorrect_final_pub) > 0 else 0.0
                    final_selected_model_accuracies_one_eps.append(acc_final_selected_pub)
                    
                    sval_errors_for_selected = error_counts_chi_valid[selected_idx_in_valid_list] if selected_idx_in_valid_list < len(error_counts_chi_valid) else "N/A"
                    log_line_repeat_vals = [
                        f'{current_epsilon_total_for_alg3:.4f}', f'{i_repeat + 1}', f'{original_selected_hyper_idx}',
                        str(selected_model_info_final['hyper_params']), f'{sval_errors_for_selected}',
                        f'{acc_final_selected_pub:.4f}', f"{selected_model_info_final['L_val']}",
                        f"{selected_model_info_final['gamma']:.3e}" if selected_model_info_final['gamma'] is not None and not np.isnan(selected_model_info_final['gamma']) else 'N/A'
                    ]
                    print('\t'.join(log_line_repeat_vals), file=logfile)
                
                if final_selected_model_accuracies_one_eps:
                    accuracies_for_plot_alg[i_eps] = np.nanmean(final_selected_model_accuracies_one_eps)
                    stds_for_plot_alg[i_eps] = np.nanstd(final_selected_model_accuracies_one_eps)
                    print(f"  Avg Acc for eps={current_epsilon_total_for_alg3:.4f} (Alg3, {alg_name_current}): {accuracies_for_plot_alg[i_eps]:.4f} +/- {stds_for_plot_alg[i_eps]:.4f}")
                logfile.flush()
            
            matrix_row_idx_alg3_fill = alg_idx + 1
            acc_matrix[matrix_row_idx_alg3_fill, :] = accuracies_for_plot_alg
            std_matrix[matrix_row_idx_alg3_fill, :] = stds_for_plot_alg
            alg_instance_name_print = config['class']() # Create instance to call name property
            print(f"{alg_instance_name_print.name}_Alg3,{','.join(map(lambda x: f'{x:.4f}' if not np.isnan(x) else 'nan', accuracies_for_plot_alg))}", file=accfile)
            print(f"{alg_instance_name_print.name}_Alg3,{','.join(map(lambda x: f'{x:.4f}' if not np.isnan(x) else 'nan', stds_for_plot_alg))}", file=stdfile)
            accfile.flush(); stdfile.flush()
        
        print('\n--- Generating Plot for Algorithm 3 Results ---')
        try:
            plt.figure(figsize=(12, 8)); # Adjusted figure size
            markers = ['o', 's', '^', 'd', 'v', '<', '>','P','X','*'] 
            plt.axhline(y=baseline_accuracy, color='grey', linestyle='--', linewidth=1.5, label=f'Non-Private Baseline ({baseline_accuracy:.3f})')
            for i_plot, alg_name_plot in enumerate(alg_name_list):
                AlgClassPlotInstance = hyper_configs[alg_name_plot]['class']()
                row_idx_plot = i_plot + 1
                acc_values_plot = acc_matrix[row_idx_plot, :]; std_values_plot = std_matrix[row_idx_plot, :]
                valid_pts_plot = ~np.isnan(acc_values_plot)
                if np.any(valid_pts_plot):
                    plt.errorbar(np.array(eps_to_run)[valid_pts_plot], acc_values_plot[valid_pts_plot], yerr=std_values_plot[valid_pts_plot],
                                 label=f'{AlgClassPlotInstance.name} (Tuned with Alg3)', fmt=f'-{markers[i_plot % len(markers)]}', capsize=3, elinewidth=1, markeredgewidth=1)
            plt.xscale('log'); plt.xlabel('Privacy Budget (ε) for Entire Algorithm 3 Process', fontsize=12); plt.ylabel('Accuracy of Selected Model', fontsize=12)
            plt.title(f'DP Accuracy vs. Epsilon using Algorithm 3 ({dataset_name} - {model_name_hint})', fontsize=14)
            plt.legend(loc='lower right', fontsize=10); plt.grid(True, which="both", ls="--", linewidth=0.5)
            min_y_on_plot = np.nanmin(acc_matrix[1:]) if acc_matrix.shape[0]>1 and np.any(~np.isnan(acc_matrix[1:])) else 0
            max_y_on_plot = np.nanmax(acc_matrix[1:]) if acc_matrix.shape[0]>1 and np.any(~np.isnan(acc_matrix[1:])) else baseline_accuracy if not np.isnan(baseline_accuracy) else 1.0

            bottom_limit = max(0, min_y_on_plot - 0.05 if not np.isnan(min_y_on_plot) else 0)
            top_limit = min(1.0, max(max_y_on_plot if not np.isnan(max_y_on_plot) else 0, baseline_accuracy if not np.isnan(baseline_accuracy) else 0) + 0.05)
            if top_limit <= bottom_limit : top_limit = bottom_limit + 0.1
            plt.ylim(bottom=bottom_limit, top=top_limit)
            
            tick_labels = [f"{e:.1f}" if e<10 else f"{e:.0f}" for e in eps_to_run]
            if len(eps_to_run) > 6: # Reduce number of ticks if too many
                tick_indices = np.linspace(0, len(eps_to_run)-1, num=min(len(eps_to_run), 6), dtype=int)
                plt.xticks(np.array(eps_to_run)[tick_indices], labels=np.array(tick_labels)[tick_indices])
            else:
                plt.xticks(eps_to_run, labels=tick_labels)
            plt.savefig(plot_filename, bbox_inches='tight'); plt.close(); print(f'Plot saved to {plot_filename}')
        except Exception as e: print(f"Error generating plot: {e}") #import traceback; traceback.print_exc()

    print('\n--- Script Finished ---')

if __name__ == '__main__':
    main()