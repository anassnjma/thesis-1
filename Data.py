import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing, fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def create_california():
    """California Housing - binary classification (above/below median price)"""
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = (data.target > np.median(data.target)).astype(int)
    return df

def create_news():
    """20 Newsgroups - binary classification (atheism vs christian)"""
    categories = ['alt.atheism', 'soc.religion.christian']
    train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers'))
    test = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers'))
    
    # Combine and vectorize
    texts = train.data + test.data
    labels = list(train.target) + list(test.target)
    
    vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
    X = vectorizer.fit_transform(texts).toarray()
    
    df = pd.DataFrame(X, columns=[f'tfidf_{i}' for i in range(X.shape[1])])
    df['target'] = labels
    return df

def create_traffic():
    """Traffic dataset - binary classification (high vs low traffic)"""
    print("Generating Traffic dataset...")
    np.random.seed(42)
    n_samples = 7909
    
    # Generate synthetic traffic data
    motorcycles = np.random.poisson(50, n_samples)  # Class 1
    pickups_vans = np.random.poisson(200, n_samples)  # Class 3
    
    # Passenger vehicles - correlated with other vehicle types
    passenger_vehicles = (motorcycles * 2.5 + pickups_vans * 1.8 +
                         np.random.normal(100, 30, n_samples)).astype(int)
    passenger_vehicles = np.maximum(passenger_vehicles, 0)  # Ensure non-negative
    
    df = pd.DataFrame({
        'motorcycles': motorcycles,
        'pickups_vans': pickups_vans,
        'passenger_vehicles': passenger_vehicles
    })
    
    # Create binary target: high traffic periods (above median passenger vehicles)
    median_traffic = np.median(passenger_vehicles)
    df['target'] = (passenger_vehicles > median_traffic).astype(int)
    
    return df

def preprocess_and_save(df, name):
    """Standardize, clip, add intercept, enforce unit ball constraint, convert labels, and save"""
    if df is None:
        print(f"Failed to create {name}")
        return
    
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Standardize and clip features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_clipped = np.clip(X_scaled, -3, 3)
    
    # Add intercept column
    intercept = np.ones((len(X_clipped), 1))
    X_with_intercept = np.hstack([intercept, X_clipped])
    
    # Enforce unit ball constraint: rescale each data point to have norm ≤ 1
    norms = np.linalg.norm(X_with_intercept, axis=1)
    # Only rescale points that exceed unit norm
    scaling_factors = np.maximum(norms, 1.0)
    X_final = X_with_intercept / scaling_factors.reshape(-1, 1)
    
    # Convert labels to {-1, +1}
    y_final = 2 * y.values - 1
    
    # Save
    os.makedirs('./data', exist_ok=True)
    np.save(f'./data/{name}_processed_x.npy', X_final)
    np.save(f'./data/{name}_processed_y.npy', y_final)
    
    print(f"{name}: {len(X_final)} samples, {X_final.shape[1]} features")
    print(f"  Max norm after unit ball constraint: {np.max(np.linalg.norm(X_final, axis=1)):.6f}")

def main():
    print("Creating datasets...")
    
    # Create and process each dataset
    datasets = {
        'california': create_california(),
        'news': create_news(),
        'traffic': create_traffic()
    }
    
    for name, df in datasets.items():
        preprocess_and_save(df, name)
    
    print("Done!")

if __name__ == "__main__":
    main()
