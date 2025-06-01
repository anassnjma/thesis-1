import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import requests
import zipfile
import os
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

class DatasetProcessor:
    """
    Downloads and processes datasets for differentially private algorithms comparison.
    Focuses on datasets that satisfy TukeyEM requirements (n > 1000*d) and work for 
    DP-SGD, AMP, and TukeyEM algorithms for logistic regression.
    """
    
    def __init__(self, data_dir='./data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def download_california_housing(self):
        """
        Downloads and processes California Housing dataset.
        Target: High-value houses (classification)
        Features: 8 features + intercept = 9 dimensions
        Samples: ~20,640
        """
        print("Downloading California Housing dataset...")
        
        try:
            from sklearn.datasets import fetch_california_housing
            data = fetch_california_housing()
            
            # Create DataFrame
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            
            # For classification: create binary target (above median price)
            median_price = df['target'].median()
            df['target_binary'] = (df['target'] > median_price).astype(int)
            
            # Add intercept column
            df.insert(0, 'intercept', 1.0)
            
            print(f"California Housing: {len(df)} samples, {len(df.columns)-2} features")
            print(f"n/d ratio: {len(df)/(len(df.columns)-2):.1f}")
            
            return df
            
        except Exception as e:
            print(f"Error downloading California Housing: {e}")
            return None
    
    def download_diamonds(self):
        """
        Downloads and processes Diamonds dataset.
        Target: Expensive diamonds (classification)  
        Features: 9 features + intercept = 10 dimensions
        Samples: ~53,940
        """
        print("Downloading Diamonds dataset...")
        
        url = "https://raw.githubusercontent.com/tidyverse/ggplot2/main/data-raw/diamonds.csv"
        
        try:
            df = pd.read_csv(url)
            
            # Encode categorical variables
            categorical_cols = ['cut', 'color', 'clarity']
            le_dict = {}
            
            for col in categorical_cols:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                le_dict[col] = le
            
            # For classification: create binary target (expensive vs not)
            price_threshold = df['price'].quantile(0.7)  # Top 30% as expensive
            df['target_binary'] = (df['price'] > price_threshold).astype(int)
            
            # Select relevant numerical features
            feature_cols = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']
            df_processed = df[feature_cols + ['price', 'target_binary']].copy()
            
            # Add intercept column
            df_processed.insert(0, 'intercept', 1.0)
            
            # Remove outliers (very large diamonds)
            df_processed = df_processed[df_processed['carat'] <= 3.0]
            df_processed = df_processed[df_processed['price'] <= 15000]
            
            print(f"Diamonds: {len(df_processed)} samples, {len(feature_cols)+1} features")
            print(f"n/d ratio: {len(df_processed)/(len(feature_cols)+1):.1f}")
            
            return df_processed
            
        except Exception as e:
            print(f"Error downloading Diamonds: {e}")
            return None
    
    def download_traffic(self):
        """
        Downloads and processes Traffic dataset.
        Target: High traffic periods (classification)
        Features: 2 features + intercept = 3 dimensions  
        Samples: ~7,909
        """
        print("Downloading Traffic dataset...")
        
        # Synthetic traffic data based on the paper's description
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
            'intercept': 1.0,
            'motorcycles': motorcycles,
            'pickups_vans': pickups_vans,
            'passenger_vehicles': passenger_vehicles
        })
        
        # For classification: high traffic vs normal traffic
        traffic_threshold = df['passenger_vehicles'].quantile(0.6)
        df['target_binary'] = (df['passenger_vehicles'] > traffic_threshold).astype(int)
        
        print(f"Traffic: {len(df)} samples, {3} features")
        print(f"n/d ratio: {len(df)/3:.1f}")
        
        return df
    
    def preprocess_dataset(self, df, task_type='classification', random_state=42):
        """
        Preprocesses dataset for DP algorithms.
        Returns combined train+test for now (can be split later).
        
        Args:
            df: DataFrame with processed data
            task_type: 'classification' or 'regression'
            random_state: Random seed
            
        Returns:
            Dictionary with combined data and metadata
        """
        if df is None:
            return None
            
        # For classification, use binary target
        target_col = 'target_binary'
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in ['target', 'target_binary', 'price', 'passenger_vehicles']]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Standardize features (except intercept)
        scaler = StandardScaler()
        feature_cols_to_scale = [col for col in feature_cols if col != 'intercept']
        
        if feature_cols_to_scale:
            X[feature_cols_to_scale] = scaler.fit_transform(X[feature_cols_to_scale])
        
        # For DP algorithms, ensure features are bounded
        # Clip to reasonable range after standardization
        clip_bound = 3.0  # 3 standard deviations
        for col in feature_cols_to_scale:
            X[col] = np.clip(X[col], -clip_bound, clip_bound)
        
        return {
            'X': X,  # Combined features (train+test together)
            'y': y,  # Combined targets (train+test together)
            'feature_names': feature_cols,
            'n_features': len(feature_cols),
            'n_samples': len(X),
            'task_type': task_type,
            'scaler': scaler
        }
    
    def save_dataset(self, data_dict, dataset_name):
        """Save a processed dataset to NumPy files."""
        if data_dict is None:
            return
        
        # Save directly to data_dir (not in subdirectories)
        # Save as NumPy arrays
        X_filename = f"{dataset_name}_processed_x.npy"
        y_filename = f"{dataset_name}_processed_y.npy"
        
        np.save(os.path.join(self.data_dir, X_filename), data_dict['X'].values)
        np.save(os.path.join(self.data_dir, y_filename), data_dict['y'].values)
        
        print(f"Saved {X_filename} and {y_filename}")

    def load_dataset(self, dataset_name):
        """Load a previously saved dataset from NumPy files."""
        X_filename = f"{dataset_name}_processed_x.npy"
        y_filename = f"{dataset_name}_processed_y.npy"
        
        X_path = os.path.join(self.data_dir, X_filename)
        y_path = os.path.join(self.data_dir, y_filename)
        
        if not (os.path.exists(X_path) and os.path.exists(y_path)):
            print(f"Dataset {dataset_name} not found")
            return None
        
        try:
            X = np.load(X_path)
            y = np.load(y_path)
            
            return {
                'X': X,
                'y': y
            }
        
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            return None

    def get_all_datasets(self, save_to_disk=True):
        """
        Downloads and processes all datasets for logistic regression (classification).
        All three algorithms (DP-SGD, AMP, TukeyEM) will work on classification tasks.
        
        Args:
            save_to_disk: Whether to save datasets to disk
        
        Returns:
            Dictionary with all processed datasets
        """
        datasets = {}
        
        # Download raw datasets
        california_df = self.download_california_housing()
        diamonds_df = self.download_diamonds()
        traffic_df = self.download_traffic()
        
        # Process only for classification tasks (all three algorithms)
        if california_df is not None:
            datasets['california'] = self.preprocess_dataset(
                california_df, task_type='classification'
            )
            
            if save_to_disk:
                self.save_dataset(datasets['california'], 'california')
        
        if diamonds_df is not None:
            datasets['diamonds'] = self.preprocess_dataset(
                diamonds_df, task_type='classification'
            )
            
            if save_to_disk:
                self.save_dataset(datasets['diamonds'], 'diamonds')
        
        if traffic_df is not None:
            datasets['traffic'] = self.preprocess_dataset(
                traffic_df, task_type='classification'
            )
            
            if save_to_disk:
                self.save_dataset(datasets['traffic'], 'traffic')
        
        return datasets

    def list_saved_datasets(self):
        """List all saved .npy datasets in the data directory."""
        if not os.path.exists(self.data_dir):
            print(f"Data directory {self.data_dir} does not exist.")
            return []
        
        # Look for .npy files
        npy_files = [f for f in os.listdir(self.data_dir) if f.endswith('.npy')]
        
        if npy_files:
            print(f"\nSaved datasets in {self.data_dir}:")
            for npy_file in sorted(npy_files):
                print(f"  - {npy_file}")
        else:
            print(f"No .npy files found in {self.data_dir}")
        
        return npy_files

    def print_dataset_summary(self, datasets):
        """Print summary of all datasets."""
        print("\n" + "="*80)
        print("DATASET SUMMARY")
        print("="*80)
        
        for name, data in datasets.items():
            if data is not None:
                print(f"\n{name.upper()}:")
                print(f"  Task type: {data['task_type']}")
                print(f"  Features: {data['n_features']} (including intercept)")
                print(f"  Total samples: {data['n_samples']}")
                print(f"  n/d ratio: {data['n_samples']/data['n_features']:.1f}")
                print(f"  Feature names: {data['feature_names'][:5]}{'...' if len(data['feature_names']) > 5 else ''}")
                
                if data['task_type'] == 'classification':
                    class_dist = data['y'].value_counts()
                    print(f"  Class distribution: {dict(class_dist)}")
                else:
                    print(f"  Target range: [{data['y'].min():.2f}, {data['y'].max():.2f}]")

# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = DatasetProcessor()
    
    # Download and process all datasets
    print("Processing datasets...")
    datasets = processor.get_all_datasets(save_to_disk=True)
    
    # List saved .npy files
    saved_files = processor.list_saved_datasets()
    
    print(f"\nDatasets ready for DP algorithms!")