import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import warnings

warnings.filterwarnings('ignore')

def preprocess_edge_iiot_dataset(
    csv_path: str,
    save_path: str = None,
    plot_distribution: bool = False,
    normalization_method: str = 'standard',  # 'standard', 'minmax', or 'both'
    handle_infinite: bool = True,
    verbose: bool = True
):
    """
    Load, clean, encode, and scale the Edge-IIoTset dataset without manual drops or feature reduction or splitting.

    Parameters:
    - csv_path: Path to the input CSV file.
    - save_path: If provided, the processed DataFrame will be saved here.
    - plot_distribution: If True, plots the distribution of attack types.
    - normalization_method: 'standard', 'minmax', or 'both'.
    - handle_infinite: If True, replaces infinite values with NaN and handles them.
    - verbose: If True, prints processing information.

    Returns:
    - X, y, label_encoder, scaler_return, feature_names
    """
    if verbose:
        print("Starting Edge-IIoTset dataset preprocessing...")
        print(f"Normalization method: {normalization_method}")

    # Load data
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        if verbose:
            print(f"Dataset loaded successfully. Shape: {df.shape}")
    except Exception as e:
        raise FileNotFoundError(f"Error loading CSV file: {e}")

    # Handle infinite values
    if handle_infinite:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        if verbose:
            inf_count = np.isinf(df.select_dtypes(include=[np.number]).values).sum()
            if inf_count > 0:
                print(f"Replaced {inf_count} infinite values with NaN")

    # Handle missing values
    initial_rows = len(df)
    missing_threshold = 0.5
    high_missing = df.columns[df.isnull().mean() > missing_threshold].tolist()
    if high_missing and verbose:
        print(f"Dropping {len(high_missing)} columns with >50% missing values: {high_missing}")
    df.drop(columns=high_missing, inplace=True)
    df.dropna(axis=0, how='any', inplace=True)
    if verbose:
        print(f"Removed {initial_rows - len(df)} rows with missing values. Remaining: {len(df)} rows")

    # Remove duplicates
    initial_rows = len(df)
    df.drop_duplicates(keep='first', inplace=True)
    if verbose:
        print(f"Removed {initial_rows - len(df)} duplicate rows")

    # Check for target column
    if 'Attack_type' not in df.columns:
        raise ValueError("Target column 'Attack_type' not found")

    # Shuffle dataset
    df = df.sample(frac=1, random_state=42)

    # Encode target
    label_encoder = LabelEncoder()
    df['Attack_type'] = label_encoder.fit_transform(df['Attack_type'])
    if verbose:
        classes = list(label_encoder.classes_)
        counts = np.bincount(df['Attack_type'])
        print(f"Target classes: {classes}")
        print(f"Class distribution: {dict(zip(classes, counts))}")

    # Separate features and target
    feature_names = [c for c in df.columns if c != 'Attack_type']
    X_df = df[feature_names].apply(pd.to_numeric, errors='coerce')
    X_df.fillna(X_df.median(), inplace=True)
    y = df['Attack_type'].values

    if verbose:
        print(f"Feature matrix shape before normalization: {X_df.shape}")

    # Normalization
    scalers = {}
    X = X_df.values
    if normalization_method == 'standard' or normalization_method == 'both':
        standard_scaler = StandardScaler()
        X = standard_scaler.fit_transform(X)
        scalers['standard'] = standard_scaler
        if verbose:
            print("Applied StandardScaler normalization")
    if normalization_method == 'minmax' or normalization_method == 'both':
        minmax_scaler = MinMaxScaler()
        X = minmax_scaler.fit_transform(X)
        scalers['minmax'] = minmax_scaler
        if verbose:
            print("Applied MinMaxScaler normalization")

    scaler_return = scalers if normalization_method == 'both' else scalers[normalization_method]

    # Optional distribution plot
    if plot_distribution:
        plt.figure(figsize=(10, 6))
        names = label_encoder.inverse_transform(np.unique(y))
        counts = np.bincount(y)
        plt.bar(names, counts)
        plt.xticks(rotation=45, ha='right')
        plt.title("Attack Type Distribution")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

    # Optional save
    if save_path:
        out_df = pd.DataFrame(X, columns=feature_names)
        out_df['Attack_type'] = y
        out_df.to_csv(save_path, index=False)
        if verbose:
            print(f"Saved processed data to {save_path}")

    if verbose:
        print("Preprocessing completed.")

    return X, y, label_encoder, scaler_return, feature_names


# main
if __name__ == "__main__":
    X, y, le, scaler, features = preprocess_edge_iiot_dataset(
        csv_path="../archive/Edge-IIoTset dataset/Selected dataset for ML and DL/ML-EdgeIIoT-dataset.csv",
        normalization_method='minmax',
        plot_distribution=True,
        save_path="Preprocessed-ML-EdgeIIoT-dataset.csv"
    )
