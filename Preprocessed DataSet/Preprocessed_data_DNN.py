import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import warnings

warnings.filterwarnings('ignore')

def preprocess_edge_iiot_dataset(
    csv_path: str,
    random_state: int = 42,
    save_path: str = None,
    plot_distribution: bool = False,
    normalization_method: str = 'standard',  # 'standard', 'minmax', or 'both'
    handle_infinite: bool = True,
    cardinality_threshold: int = 50,  # Reduced threshold for better memory management
    verbose: bool = True
):
    """
    Load, clean, encode, scale the Edge-IIoTset dataset,
    with one‐hot encoding for categorical INPUT features only.
    TARGET variables (Attack_type, Attack_label) use Label Encoding.

    Parameters:
    - csv_path: Path to the input CSV file.
    - random_state: Seed for reproducibility.
    - save_path: If provided, the processed DataFrame will be saved here.
    - plot_distribution: If True, plots the distribution of attack types.
    - normalization_method: 'standard', 'minmax', or 'both'.
    - handle_infinite: If True, replaces infinite values with NaN and handles them.
    - cardinality_threshold: Maximum unique‐value count allowed for one‐hot.
    - verbose: If True, prints processing information.

    Returns:
    - X, y_attack_type, y_attack_label, label_encoders, scaler_return, feature_names
    """
    if verbose:
        print("Starting Edge-IIoTset dataset preprocessing...")
        print(f"Normalization method: {normalization_method}")
        print(f"High‐cardinality threshold: {cardinality_threshold} unique values\n")

    # ----------------------------
    # 1) Load data
    # ----------------------------
    
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        if verbose:
            print(f"Dataset loaded successfully. Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}\n")
    except Exception as e:
        raise FileNotFoundError(f"Error loading CSV file: {e}")

    # ----------------------------
    # 2) Handle infinite values
    # ----------------------------
    
    if handle_infinite:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        if verbose:
            numeric_vals = df.select_dtypes(include=[np.number]).values
            inf_count = np.isinf(numeric_vals).sum()
            if inf_count > 0:
                print(f"Warning: still found {inf_count} infinite values after replacing.\n")

    # ----------------------------
    # 3) Drop columns/rows with too many missing values
    # ----------------------------
    
    initial_rows = len(df)
    missing_threshold = 0.5
    high_missing_cols = df.columns[df.isnull().mean() > missing_threshold].tolist()
    if high_missing_cols and verbose:
        print(f"Dropping {len(high_missing_cols)} columns with >50% missing values:\n  {high_missing_cols}\n")
    df.drop(columns=high_missing_cols, inplace=True)

    df.dropna(axis=0, how='any', inplace=True)
    if verbose:
        print(f"Dropped rows with any NaN. Removed {initial_rows - len(df)} rows → Remaining: {len(df)} rows\n")

    # ----------------------------
    # 4) Remove duplicates
    # ----------------------------
    initial_rows = len(df)
    df.drop_duplicates(keep='first', inplace=True)
    if verbose:
        print(f"Removed {initial_rows - len(df)} duplicate rows → Remaining: {len(df)} rows\n")

    # ----------------------------
    # 5) Identify target columns
    # ----------------------------
    
    target_columns = []
    if 'Attack_type' in df.columns:
        target_columns.append('Attack_type')
    if 'Attack_label' in df.columns:
        target_columns.append('Attack_label')
    
    if not target_columns:
        raise ValueError("Neither 'Attack_type' nor 'Attack_label' found in dataset. Cannot proceed.")
    
    if verbose:
        print(f"Target columns found: {target_columns}\n")

    # ----------------------------
    # 6) Shuffle dataset
    # ----------------------------
    
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # ----------------------------
    # 7) Encode target variables with LabelEncoder (NOT One-Hot)
    # ----------------------------
    
    label_encoders = {}
    
    for target_col in target_columns:
        le = LabelEncoder()
        df[target_col + '_encoded'] = le.fit_transform(df[target_col])
        label_encoders[target_col] = le
        
        if verbose:
            classes = list(le.classes_)
            counts = np.bincount(df[target_col + '_encoded'])
            class_dist = dict(zip(classes, counts))
            print(f"{target_col} classes: {classes}")
            print(f"{target_col} distribution: {class_dist}\n")

    # ----------------------------
    # 8) Separate features from targets
    # ----------------------------
    
    # Get all columns except original and encoded target columns
    exclude_cols = target_columns + [col + '_encoded' for col in target_columns]
    feature_names = [c for c in df.columns if c not in exclude_cols]
    X_df = df[feature_names].copy()
    
    if verbose:
        print(f"Input features shape: {X_df.shape}")
        print(f"Number of input features: {len(feature_names)}\n")

    # ----------------------------
    # 9) One‐Hot encode ONLY categorical INPUT features
    # ----------------------------
    
    # Identify categorical columns in INPUT features only
    cat_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
    if verbose and cat_cols:
        print(f"Categorical INPUT columns found: {cat_cols}")
        for col in cat_cols:
            nunique = X_df[col].nunique()
            print(f"  {col}: {nunique} unique values")
        print()

    # Filter out high-cardinality categorical columns
    high_card_cols = []
    safe_cat_cols = []
    
    for col in cat_cols:
        nunique = X_df[col].nunique()
        if nunique > cardinality_threshold:
            high_card_cols.append(col)
        else:
            safe_cat_cols.append(col)

    if high_card_cols and verbose:
        print(f"Dropping {len(high_card_cols)} high‐cardinality INPUT columns (>{cardinality_threshold} unique values):")
        for col in high_card_cols:
            print(f"  {col}: {X_df[col].nunique()} unique values")
        print()
        X_df.drop(columns=high_card_cols, inplace=True)

    # Apply one-hot encoding to remaining categorical INPUT features
    if safe_cat_cols:
        if verbose:
            print(f"Applying one-hot encoding to {len(safe_cat_cols)} categorical INPUT columns:")
            for col in safe_cat_cols:
                print(f"  {col}: {X_df[col].nunique()} unique values")
            print()
        
        try:
            X_df = pd.get_dummies(X_df, columns=safe_cat_cols, drop_first=True)
            if verbose:
                print(f"One-hot encoding successful. New shape: {X_df.shape}\n")
        except MemoryError:
            if verbose:
                print("MemoryError during one-hot encoding. Dropping categorical columns.\n")
            X_df.drop(columns=safe_cat_cols, inplace=True)
    else:
        if verbose:
            print("No categorical INPUT columns to one-hot encode.\n")

    # ----------------------------
    # 10) Convert everything to numeric & handle NaNs
    # ----------------------------
    
    X_df = X_df.apply(pd.to_numeric, errors='coerce')
    
    # Fill NaNs with median for numeric columns
    numeric_cols = X_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if X_df[col].isnull().any():
            median_val = X_df[col].median()
            X_df[col].fillna(median_val, inplace=True)

    # Update feature names after one-hot encoding
    feature_names = X_df.columns.tolist()
    if verbose:
        print(f"Final feature matrix shape: {X_df.shape}")
        print(f"Final number of features: {len(feature_names)}\n")

    # ----------------------------
    # 11) Normalization
    # ----------------------------
    
    scalers = {}
    X = X_df.values

    if normalization_method in ('standard', 'both'):
        standard_scaler = StandardScaler()
        X = standard_scaler.fit_transform(X)
        scalers['standard'] = standard_scaler
        if verbose:
            print("Applied StandardScaler normalization.")

    if normalization_method in ('minmax', 'both'):
        minmax_scaler = MinMaxScaler()
        X = minmax_scaler.fit_transform(X)
        scalers['minmax'] = minmax_scaler
        if verbose:
            print("Applied MinMaxScaler normalization.")

    if normalization_method == 'both':
        scaler_return = scalers
    else:
        scaler_return = scalers.get(normalization_method)

    # ----------------------------
    # 12) Prepare target variables
    # ----------------------------
    
    y_attack_type = df['Attack_type_encoded'].values if 'Attack_type' in target_columns else None
    y_attack_label = df['Attack_label_encoded'].values if 'Attack_label' in target_columns else None

    # ----------------------------
    # 13) Optional: Plot distribution
    # ----------------------------
    
    if plot_distribution and y_attack_type is not None:
        plt.figure(figsize=(12, 6))
        
        if y_attack_label is not None:
            plt.subplot(1, 2, 1)
        
        # Plot Attack_type distribution
        attack_type_classes = label_encoders['Attack_type'].inverse_transform(np.unique(y_attack_type))
        attack_type_counts = np.bincount(y_attack_type)
        plt.bar(range(len(attack_type_classes)), attack_type_counts)
        plt.xticks(range(len(attack_type_classes)), attack_type_classes, rotation=45, ha='right')
        plt.title("Attack Type Distribution")
        plt.ylabel("Count")
        
        if y_attack_label is not None:
            plt.subplot(1, 2, 2)
            # Plot Attack_label distribution
            attack_label_classes = label_encoders['Attack_label'].inverse_transform(np.unique(y_attack_label))
            attack_label_counts = np.bincount(y_attack_label)
            plt.bar(range(len(attack_label_classes)), attack_label_counts)
            plt.xticks(range(len(attack_label_classes)), attack_label_classes, rotation=45, ha='right')
            plt.title("Attack Label Distribution")
            plt.ylabel("Count")
        
        plt.tight_layout()
        plt.show()

    # ----------------------------
    # 14) Optional: Save processed data
    # ----------------------------
    
    if save_path:
        out_df = pd.DataFrame(X, columns=feature_names)
        if y_attack_type is not None:
            out_df['Attack_type_encoded'] = y_attack_type
        if y_attack_label is not None:
            out_df['Attack_label_encoded'] = y_attack_label
        out_df.to_csv(save_path, index=False)
        if verbose:
            print(f"Saved processed data to `{save_path}`\n")

    if verbose:
        print("Preprocessing completed successfully!\n")
        print("Summary:")
        print(f"- Input features: {X.shape}")
        print(f"- Attack_type target: {'Available' if y_attack_type is not None else 'Not found'}")
        print(f"- Attack_label target: {'Available' if y_attack_label is not None else 'Not found'}")

    return X, y_attack_type, y_attack_label, label_encoders, scaler_return, feature_names


# =======================
# Example usage
# =======================
if __name__ == "__main__":
    csv_path = "../../archive/Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv"  # Update with your actual path

    X, y_type, y_label, encoders, scaler, features = preprocess_edge_iiot_dataset(
        csv_path=csv_path,
        normalization_method='minmax',
        plot_distribution=True,
        save_path="Preprocessed-DNN-EdgeIIoT-dataset.csv",
        cardinality_threshold=50,  # Reduced for better memory management
        verbose=True
    )
    
    print("Preprocessing complete!")
    print(f"Features shape: {X.shape}")
    if y_type is not None:
        print(f"Attack_type shape: {y_type.shape}, unique classes: {len(np.unique(y_type))}")
    if y_label is not None:
        print(f"Attack_label shape: {y_label.shape}, unique classes: {len(np.unique(y_label))}")