import os
import sys
import csv
import numpy as np
import sklearn.linear_model
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from multiprocessing import Pool 
from itertools import product
from Algorithms.DPPSGD import DPSGDLR #output perturbation SGD
from Algorithms.AMP import ApproximateMinimaPerturbationLR #approximate minima perturbation 
from Algorithms.TukeyEM import DPTukey #Tukey depth exponential mechanism
from Algorithms.DSGD import GradientPerturbationDPSGD #gradient perturbation SGD

##########################################################################
# Configuration
##########################################################################
DATASET_LOCATION = '/Users/anass/Desktop/Thesis/Code/Thesis_code/datasets/data'
AVAILABLE_DATASETS = ['california', 'diamonds', 'traffic', 'adult'] 
NUM_REPEATS = 3
CORES = 4 
EPSILON_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
MAIN_SEED = 42
MIN_SUBSET_SIZE = 50 

##########################################################################
# Helper Functions
##########################################################################
def build_binary_labels(y, positive_class_label=1):
    y_arr = np.asarray(y).ravel()
    unique_y = set(np.unique(y_arr))
    if unique_y.issubset({-1, 1}):
        return np.where(y_arr == 1, 1, -1).astype(int)
    return np.where(y_arr == positive_class_label, 1, -1).astype(int)

def create_directory(directory_name):
    try:
        os.makedirs(directory_name, exist_ok=True)
    except OSError:
        pass

def dict_product(param_dict):
    keys = param_dict.keys()
    values = param_dict.values()
    for combination in product(*values):
        yield dict(zip(keys, combination))

##########################################################################
# Dataset Loading
##########################################################################
def load_dataset(dataset_name):
    if dataset_name not in AVAILABLE_DATASETS:
        raise ValueError(f"Dataset '{dataset_name}' not available")
    
    x_path = os.path.join(DATASET_LOCATION, f"{dataset_name}_processed_x.npy")
    y_path = os.path.join(DATASET_LOCATION, f"{dataset_name}_processed_y.npy")
    
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        raise FileNotFoundError(f"Dataset files not found for '{dataset_name}'")
    
    X = np.load(x_path).astype(float)
    y_original = np.load(y_path).astype(int).ravel()
    
    unique_labels_orig = np.unique(y_original)
    if 1 in unique_labels_orig:
        positive_class_val = 1
    elif 0 in unique_labels_orig and len(unique_labels_orig) == 2:
        positive_class_val = np.max(unique_labels_orig)
    else:
        positive_class_val = np.max(unique_labels_orig) if unique_labels_orig.size > 0 else 1
    
    y_binary = build_binary_labels(y_original, positive_class_label=positive_class_val)
    return X, y_binary

##########################################################################
# Model Evaluation
##########################################################################
def predict_binary(theta, X):
    if theta is None or len(theta) == 0:
        return np.ones(X.shape[0], dtype=int) * -1 
    scores = X.dot(theta)
    return np.where(scores >= 0, 1, -1).astype(int)

def evaluate_accuracy(theta, X, y_true):
    if theta is None: 
        return 0.0
    y_pred = predict_binary(theta, X)
    return np.mean(y_pred == y_true)

def count_errors(theta, X, y_true):
    if theta is None: 
        return X.shape[0] if X.shape[0] > 0 else 0
    if X.shape[0] == 0: 
        return 0
    y_pred = predict_binary(theta, X)
    return np.sum(y_pred != y_true)

##########################################################################
# Training Tasks
##########################################################################
def train_model_task(args):
    alg_class, X_train, y_train = args['alg_class'], args['X_train'], args['y_train']
    epsilon, delta = args['epsilon'], args['delta']
    hyperparams, random_state = args['hyperparams'], args['random_state']
    
    try:
        algorithm = alg_class() 
        theta, _, _ = algorithm.run_classification(
            x=X_train, y=y_train, 
            epsilon=epsilon, delta=delta,
            random_state=random_state, **hyperparams)
        success = theta is not None and not np.allclose(theta, 0.0, atol=1e-12)
        return {'theta': theta if success else None, 'hyperparams': hyperparams, 'success': success}
    except Exception:
        return {'theta': None, 'hyperparams': hyperparams, 'success': False}

##########################################################################
# Hyperparameter Configuration
##########################################################################
def get_hyperparameter_configs(dataset_name):
    base_lambda = [1e-4, 1e-3, 1e-2]
    base_epochs = [20, 50]
    base_batch_sizes = [32, 64]
    base_L = [0.5, 1.0, 2.0]
    
    # Simplified configurations
    if dataset_name == "california":
        tukey_m = [5, 8]
    elif dataset_name == "diamonds":
        tukey_m = [8, 12]
    elif dataset_name == "traffic":
        tukey_m = [3, 5]
    else: 
        tukey_m = [5, 8]
    
    return {
        'DPSGD': {
            'class': DPSGDLR,
            'params': {
                'lambda_param': base_lambda, 
                'num_epochs': base_epochs,
                'batch_size': base_batch_sizes, 
                'L': base_L
            }
        },
        'AMP': {
            'class': ApproximateMinimaPerturbationLR,
            'params': {
                'L': [1.0, 2.0],                    
                'l2_constraint': [None, 5.0],       
                'eps_out_frac': [0.01, 0.05],         
            }
        },
        'TukeyEM': {
            'class': DPTukey,
            'params': {
                'm': tukey_m,
                'L_data_clipping': [None],
                'logreg_C': [1.0],
                'logreg_fit_intercept': [False],
                'ptr_t_divisor': [8, 12],
                'exp_mech_t_divisor': [8, 12],
                'logreg_solver': ['liblinear'],
                'debug': [False]
            }
        },
        'GradientPerturbationDPSGD': {
            'class': GradientPerturbationDPSGD,
            'params': {
                'num_iters': [50, 100, 200],
                'learning_rate': [0.001, 0.01, 0.1], 
                'L': [0.5, 1.0, 2.0],
                'minibatch_size': [32, 64, 128],
                'l2_constraint': [None, 1.0, 5.0],
                'lambda_param': [0, 1e-4, 1e-3, 1e-2]
            }
        }
    }

##########################################################################
# Private Hyperparameter Tuning
##########################################################################
def private_hyperparameter_tuning(alg_class, param_grid, X_tuning, y_tuning, 
                                  epsilon_tuning, delta, random_state_base):
    param_combinations = list(dict_product(param_grid))
    n_candidates = len(param_combinations)

    if n_candidates == 0:
        return {}, X_tuning.shape[0]

    # Split epsilon budget
    epsilon_training = epsilon_tuning / 2.0
    epsilon_selection = epsilon_tuning / 2.0

    # Split data
    n_tuning = X_tuning.shape[0]
    num_splits = n_candidates + 1
    subset_size = n_tuning // num_splits

    if subset_size < MIN_SUBSET_SIZE:
        return param_combinations[0] if param_combinations else {}, n_tuning

    # Prepare data splits
    rng = np.random.default_rng(random_state_base)
    indices = rng.permutation(n_tuning)
    
    # Training subsets for each candidate
    train_subsets_X, train_subsets_y = [], []
    for i in range(n_candidates):
        start_idx = i * subset_size
        end_idx = (i + 1) * subset_size
        subset_indices = indices[start_idx:end_idx]
        train_subsets_X.append(X_tuning[subset_indices])
        train_subsets_y.append(y_tuning[subset_indices])
    
    # Validation set
    val_indices = indices[n_candidates * subset_size:]
    X_val, y_val = X_tuning[val_indices], y_tuning[val_indices]

    if X_val.shape[0] == 0:
        return param_combinations[0] if param_combinations else {}, n_tuning

    # Train candidates
    train_tasks = [{
        'alg_class': alg_class, 
        'X_train': train_subsets_X[i],
        'y_train': train_subsets_y[i],
        'epsilon': epsilon_training, 
        'delta': delta,
        'hyperparams': params, 
        'random_state': random_state_base + i + 1
    } for i, params in enumerate(param_combinations)]

    with Pool(CORES) as pool:
        results = pool.map(train_model_task, train_tasks)

    # Evaluate on validation set
    error_counts = []
    for result in results:
        theta = result['theta']
        errors = count_errors(theta, X_val, y_val)
        error_counts.append(errors)

    # Exponential mechanism for selection
    scores = -np.array(error_counts, dtype=float)
    sensitivity = 1.0
    
    if scores.size == 0:
        return param_combinations[0] if param_combinations else {}, n_tuning

    # Numerical stability
    max_score = np.max(scores)
    stable_scores = scores - max_score 
    exp_weights = np.exp(epsilon_selection * stable_scores / (2 * sensitivity))

    if np.sum(exp_weights) == 0:
        probabilities = np.ones(n_candidates) / n_candidates
    else:
        probabilities = exp_weights / np.sum(exp_weights)

    rng_select = np.random.default_rng(random_state_base + n_candidates + 101)
    selected_idx = rng_select.choice(n_candidates, p=probabilities)
    
    return param_combinations[selected_idx], error_counts[selected_idx]

##########################################################################
# Main Experiment
##########################################################################
def run_experiment(dataset_name, algorithm_names):
    print(f"Running experiment: {dataset_name}, Algorithms: {', '.join(algorithm_names)}")

    try:
        X_full, y_full = load_dataset(dataset_name)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None

    # Main train/test split
    rng = np.random.default_rng(MAIN_SEED)
    n_total = X_full.shape[0]
    indices = rng.permutation(n_total)
    split_idx = int(0.8 * n_total)
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]
    X_train_all, y_train_all = X_full[train_indices], y_full[train_indices]
    X_test, y_test = X_full[test_indices], y_full[test_indices]

    if X_train_all.shape[0] == 0:
        print("No training data available")
        return None
        
    delta = 1.0 / (X_train_all.shape[0]**2)
    hyperparam_configs = get_hyperparameter_configs(dataset_name)
    results = {}
    results_dir = f"./results_{dataset_name}_seed{MAIN_SEED}"
    create_directory(results_dir)

    for alg_name in algorithm_names:
        if alg_name not in hyperparam_configs:
            continue
            
        print(f"\nProcessing {alg_name}")
        alg_config = hyperparam_configs[alg_name]
        alg_class, param_grid = alg_config['class'], alg_config['params']
        alg_results = {}

        for epsilon_total in EPSILON_VALUES:
            repeat_accuracies = []

            for repeat in range(NUM_REPEATS):
                seed = MAIN_SEED + repeat + EPSILON_VALUES.index(epsilon_total) * NUM_REPEATS
                
                # Split epsilon budget
                epsilon_tuning = epsilon_total / 2.0
                epsilon_final = epsilon_total / 2.0

                # Split training data for tuning vs final training
                rng_repeat = np.random.default_rng(seed)
                n_train = X_train_all.shape[0]
                train_indices_repeat = rng_repeat.permutation(n_train)
                tuning_split = int(0.75 * n_train)
                
                tuning_indices = train_indices_repeat[:tuning_split]
                final_indices = train_indices_repeat[tuning_split:]
                
                X_tuning = X_train_all[tuning_indices]
                y_tuning = y_train_all[tuning_indices]
                X_final = X_train_all[final_indices]
                y_final = y_train_all[final_indices]

                if X_tuning.shape[0] < MIN_SUBSET_SIZE * 2 or X_final.shape[0] < MIN_SUBSET_SIZE:
                    repeat_accuracies.append(0.0)
                    continue

                # Private hyperparameter tuning
                selected_params, _ = private_hyperparameter_tuning(
                    alg_class, param_grid, X_tuning, y_tuning,
                    epsilon_tuning, delta, seed + 1000
                )

                # Train final model
                final_theta = None
                if selected_params:
                    try:
                        algorithm = alg_class()
                        final_theta, _, _ = algorithm.run_classification(
                            x=X_final, y=y_final,
                            epsilon=epsilon_final, delta=delta,
                            random_state=seed + 2000,
                            **selected_params
                        )
                        if final_theta is None or np.allclose(final_theta, 0.0, atol=1e-12):
                            final_theta = None 
                    except Exception:
                        final_theta = None

                # Evaluate
                test_accuracy = evaluate_accuracy(final_theta, X_test, y_test)
                repeat_accuracies.append(test_accuracy)

            mean_acc = np.mean(repeat_accuracies) if repeat_accuracies else 0.0
            std_acc = np.std(repeat_accuracies) if repeat_accuracies else 0.0
            alg_results[epsilon_total] = {
                'mean': mean_acc, 
                'std': std_acc, 
                'individual': repeat_accuracies
            }
            print(f"  ε={epsilon_total}: {mean_acc:.4f} ± {std_acc:.4f}")
            
        results[alg_name] = alg_results
    
    save_results(results, dataset_name, results_dir)
    plot_results(results, dataset_name, results_dir, X_full, y_full)
    return results

def save_results(results, dataset_name, results_dir):
    csv_filename = os.path.join(results_dir, f"{dataset_name}_results.csv")
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = ['Algorithm', 'Epsilon'] + \
                     [f'Repeat_{i+1}' for i in range(NUM_REPEATS)] + \
                     ['Mean', 'Std']
            writer.writerow(header)
            
            for alg_name, alg_results in results.items():
                for epsilon, data in alg_results.items():
                    individual = data.get('individual', [])
                    padded = (list(individual) + [0.0] * NUM_REPEATS)[:NUM_REPEATS]
                    row = [alg_name, epsilon] + \
                          [f"{acc:.4f}" for acc in padded] + \
                          [f"{data.get('mean', 0.0):.4f}", f"{data.get('std', 0.0):.4f}"]
                    writer.writerow(row)
        print(f"Results saved to: {csv_filename}")
    except IOError as e:
        print(f"Error saving results: {e}")

def plot_results(results, dataset_name, results_dir, X_full, y_full):
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(results)))
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, (alg_name, alg_results) in enumerate(results.items()):
        epsilons = sorted(alg_results.keys())
        means = [alg_results[eps]['mean'] for eps in epsilons]
        stds = [alg_results[eps]['std'] for eps in epsilons]
        
        plt.errorbar(epsilons, means, yerr=stds, 
                     label=alg_name, color=colors[i % len(colors)], 
                     marker=markers[i % len(markers)], 
                     capsize=5, linewidth=2, markersize=6)

    # Add baseline if possible
    try:
        rng = np.random.default_rng(MAIN_SEED)
        n_total = X_full.shape[0]
        indices = rng.permutation(n_total)
        split_idx = int(0.8 * n_total)
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        
        X_train_base = X_full[train_idx]
        y_train_base = np.where(y_full[train_idx] == 1, 1, 0)
        X_test_base = X_full[test_idx] 
        y_test_base = np.where(y_full[test_idx] == 1, 1, 0)

        if X_train_base.shape[0] > 0:
            baseline = sklearn.linear_model.LogisticRegression(
                solver='liblinear', C=1.0, random_state=MAIN_SEED, max_iter=2000)
            baseline.fit(X_train_base, y_train_base)
            baseline_acc = baseline.score(X_test_base, y_test_base)
            plt.axhline(y=baseline_acc, color='gray', linestyle='--', 
                        label=f'Non-Private Baseline ({baseline_acc:.3f})')
    except Exception:
        pass

    plt.xscale('log')
    plt.xlabel('Privacy Budget (ε)')
    plt.ylabel('Test Accuracy')
    plt.title(f'Accuracy vs Privacy Budget - {dataset_name.title()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)
    
    plot_filename = os.path.join(results_dir, f"{dataset_name}_plot.png")
    try:
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_filename}")
    except IOError as e:
        print(f"Error saving plot: {e}")
    plt.close()

def main():
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <dataset> <algorithm|ALL>")
        print(f"Datasets: {', '.join(AVAILABLE_DATASETS)}")
        print(f"Algorithms: DPSGD, AMP, TukeyEM, ALL")
        sys.exit(1)
    
    dataset_name = sys.argv[1].lower()
    algorithm_choice = sys.argv[2].upper()

    if dataset_name not in AVAILABLE_DATASETS:
        print(f"Unknown dataset: {dataset_name}")
        sys.exit(1)
    
    if algorithm_choice == 'ALL':
        algorithms = ['DPSGD', 'AMP', 'TukeyEM']
    elif algorithm_choice in ['DPSGD', 'AMP', 'TUKEYEM']:
        algorithms = ['TukeyEM' if algorithm_choice == 'TUKEYEM' else algorithm_choice]
    else:
        print(f"Unknown algorithm: {algorithm_choice}")
        sys.exit(1)

    print(f"Dataset: {dataset_name}")
    print(f"Algorithms: {', '.join(algorithms)}")
    print(f"Epsilon values: {EPSILON_VALUES}")
    print(f"Repeats: {NUM_REPEATS}")

    results = run_experiment(dataset_name, algorithms)
    
    if results:
        print("\nFinal Results:")
        for alg_name, alg_results in results.items():
            print(f"\n{alg_name}:")
            for eps, data in sorted(alg_results.items()):
                print(f"  ε={eps}: {data['mean']:.4f} ± {data['std']:.4f}")
    else:
        print("Experiment failed")

if __name__ == '__main__':
    main()