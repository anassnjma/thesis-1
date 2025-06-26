import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from itertools import product
from sklearn.linear_model import LogisticRegression
from AMP import AMP
from DPSGD import DPSGD
from DPPSGD import DPPSGD

# Config
DATASET_LOCATION = '/Users/anass/Desktop/new/data'
AVAILABLE_DATASETS = ['california', 'news', 'traffic']
NUM_REPEATS = 5
CORES = 4
EPSILON_VALUES = [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
SEED = 42

# Algorithm wrappers
def amp_wrapper(D, epsilon, delta, **kwargs):
    return AMP(D, epsilon, delta, **kwargs)

def dpsgd_wrapper(D, epsilon, delta, **kwargs):
    return DPSGD(D, epsilon, delta, **kwargs)

def dppsgd_wrapper(D, epsilon, delta, **kwargs):
    return DPPSGD(D, **kwargs, epsilon=epsilon)

ALGORITHMS = {
    'AMP': amp_wrapper, 'DPSGD': dpsgd_wrapper, 'DPPSGD': dppsgd_wrapper
}

# Parameter grids
PARAM_GRIDS = {
    'AMP': {'L': [1.0], 'eps_2_frac': [0.4, 0.7], 'eps_3_frac': [0.4, 0.7], 'delta_2_frac': [0.4,0.5]},
    'DPSGD': {'L': [0.5, 1.0, 2.0], 'T': [100, 200, 400], 'k': [32, 64], 'learning_rate': [0.001, 0.01, 0.1, 1]},
    'DPPSGD': {'k': [3, 5], 'eta': [0.01, 0.05], 'batch_size': [32], 'reg_lambda': [0.01]}
    }

# Utilities
def load_dataset(dataset_name):
    X = np.load(os.path.join(DATASET_LOCATION, f"{dataset_name}_processed_x.npy")).astype(float)
    y = np.load(os.path.join(DATASET_LOCATION, f"{dataset_name}_processed_y.npy")).astype(int).ravel()
    unique_labels = np.unique(y)
    positive_class = 1 if 1 in unique_labels else np.max(unique_labels)
    return X, np.where(y == positive_class, 1, -1).astype(int)

def evaluate_accuracy(theta, X, y):
    if theta is None: return 0.0
    return np.mean(np.where(X.dot(theta) >= 0, 1, -1) == y)

def count_errors(theta, X, y):
    if theta is None: return X.shape[0]
    return np.sum(np.where(X.dot(theta) >= 0, 1, -1) != y)

def dict_product(d):
    return [dict(zip(d.keys(), v)) for v in product(*d.values())]

# Training
def train_model(args):
    alg_func, X, y, eps, delta, params, seed = args
    np.random.seed(seed)
    theta = alg_func(D=(X, y), epsilon=eps, delta=delta, **params)
    return theta if theta is not None and not np.allclose(theta, 0, atol=1e-12) else None

def private_tune(alg_func, param_grid, X_tune, y_tune, eps_tune, delta, seed):
    param_combos = dict_product(param_grid)
    n_candidates = len(param_combos)
    if n_candidates == 0: return {}
    
    n = X_tune.shape[0]
    subset_size = n // (n_candidates + 1)
    if subset_size < 50: return param_combos[0]
    
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    
    # Train candidates
    tasks = []
    for i, params in enumerate(param_combos):
        start, end = i * subset_size, (i + 1) * subset_size
        subset_idx = indices[start:end]
        tasks.append((alg_func, X_tune[subset_idx], y_tune[subset_idx], eps_tune/2, delta, params, seed + i))
    
    with Pool(CORES) as pool:
        thetas = pool.map(train_model, tasks)
    
    # Validate and select
    val_idx = indices[n_candidates * subset_size:]
    if len(val_idx) == 0: return param_combos[0]
    
    X_val, y_val = X_tune[val_idx], y_tune[val_idx]
    errors = [count_errors(theta, X_val, y_val) for theta in thetas]
    scores = -np.array(errors, dtype=float)
    weights = np.exp((eps_tune/2) * (scores - np.max(scores)) / 2)
    probs = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(n_candidates) / n_candidates
    
    selected = rng.choice(n_candidates, p=probs)
    return param_combos[selected]

# Main Experiment
def run_experiment(dataset_name, algorithm_names):
    print(f"Dataset: {dataset_name}, Algorithms: {algorithm_names}")
    
    # Load and split data
    X_full, y_full = load_dataset(dataset_name)
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(X_full))
    split = int(0.8 * len(X_full))
    
    X_train, y_train = X_full[indices[:split]], y_full[indices[:split]]
    X_test, y_test = X_full[indices[split:]], y_full[indices[split:]]
    
    delta = 1.0 / (len(X_train)**2)
    results = {}
    
    for alg_name in algorithm_names:
        if alg_name not in ALGORITHMS: continue
        
        alg_func = ALGORITHMS[alg_name]
        param_grid = PARAM_GRIDS[alg_name]
        alg_results = {}
        
        for epsilon_total in EPSILON_VALUES:
            accuracies = []
            
            for repeat in range(NUM_REPEATS):
                seed = SEED + repeat + len(EPSILON_VALUES) * repeat
                
                # Split data for tuning/training
                rng_r = np.random.default_rng(seed)
                train_idx = rng_r.permutation(len(X_train))
                tune_split = int(0.75 * len(X_train))
                
                X_tune = X_train[train_idx[:tune_split]]
                y_tune = y_train[train_idx[:tune_split]]
                X_final = X_train[train_idx[tune_split:]]
                y_final = y_train[train_idx[tune_split:]]
                
                if len(X_tune) < 100 or len(X_final) < 50:
                    accuracies.append(0.0)
                    continue
                
                # Tune and train
                selected_params = private_tune(alg_func, param_grid, X_tune, y_tune, epsilon_total/2, delta, seed)
                
                np.random.seed(seed + 1000)
                final_theta = None
                if selected_params:
                    final_theta = alg_func(D=(X_final, y_final), epsilon=epsilon_total/2, delta=delta, **selected_params)
                
                accuracy = evaluate_accuracy(final_theta, X_test, y_test)
                accuracies.append(accuracy)
            
            alg_results[epsilon_total] = {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies),
                'individual': accuracies
            }
        
        results[alg_name] = alg_results
    
    save_results(results, dataset_name, X_full, y_full)
    return results

# Save Results
def save_results(results, dataset_name, X_full, y_full):
    os.makedirs(f"./results_{dataset_name}", exist_ok=True)
    
    # CSV
    with open(f"./results_{dataset_name}/{dataset_name}_results.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Algorithm', 'Epsilon', 'Mean', 'Std'])
        for alg, alg_res in results.items():
            for eps, data in alg_res.items():
                writer.writerow([alg, eps, f"{data['mean']:.4f}", f"{data['std']:.4f}"])
    
    # Plot
    plt.figure(figsize=(10, 6))
    colors = ['red', 'blue', 'green']
    
    for i, (alg, alg_res) in enumerate(results.items()):
        epsilons = sorted(alg_res.keys())
        means = [alg_res[eps]['mean'] for eps in epsilons]
        stds = [alg_res[eps]['std'] for eps in epsilons]
        plt.errorbar(epsilons, means, yerr=stds, label=alg, color=colors[i], 
                    marker='o', capsize=3, linewidth=2, markersize=6)
    
    # Non-private baseline
    rng = np.random.default_rng(SEED)
    indices = rng.permutation(len(X_full))
    split = int(0.8 * len(X_full))
    X_tr, y_tr = X_full[indices[:split]], y_full[indices[:split]]
    X_te, y_te = X_full[indices[split:]], y_full[indices[split:]]
    baseline = LogisticRegression(random_state=SEED, max_iter=1000)
    baseline.fit(X_tr, np.where(y_tr == 1, 1, 0))
    baseline_acc = baseline.score(X_te, np.where(y_te == 1, 1, 0))
    plt.axhline(y=baseline_acc, color='gray', linestyle='--', label='Non-private baseline', linewidth=2)

    plt.xscale('log')
    plt.xlabel('Privacy Budget (ε)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'{dataset_name.title()} - DP Algorithms', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"./results_{dataset_name}/{dataset_name}_plot.png", dpi=300, bbox_inches='tight')
    plt.close()

# CLI
def main():
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <dataset> <algorithm|ALL>")
        print(f"Available algorithms: {list(ALGORITHMS.keys())}")
        sys.exit(1)
    
    dataset_name = sys.argv[1].lower()
    algorithm_choice = sys.argv[2].upper()
    
    if dataset_name not in AVAILABLE_DATASETS:
        print(f"Unknown dataset: {dataset_name}")
        sys.exit(1)
    
    if algorithm_choice == 'ALL':
        algorithms = list(ALGORITHMS.keys())
    elif algorithm_choice in ALGORITHMS:
        algorithms = [algorithm_choice]
    else:
        print(f"Unknown algorithm: {algorithm_choice}")
        sys.exit(1)
    
    results = run_experiment(dataset_name, algorithms)
    
    print("\\nResults:")
    for alg, alg_res in results.items():
        print(f"{alg}:")
        for eps, data in sorted(alg_res.items()):
            print(f"  ε={eps}: {data['mean']:.3f} ± {data['std']:.3f}")

if __name__ == '__main__':
    main()
