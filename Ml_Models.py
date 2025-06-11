import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, validation_curve, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path):
    """
    Load EdgeIIoTset dataset and prepare for training
    
    Args:
        file_path (str): Path to the EdgeIIoTset CSV file
    
    Returns:
        tuple: X_train, X_test, y_train, y_test, feature_names, label_encoder, scaler
    """
    # Load the dataset
    print("Loading EdgeIIoTset dataset...")
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    
    # Display basic information about the dataset
    print(f"\nDataset Info:")
    print(f"Total samples: {len(df)}")
    print(f"Total features: {df.shape[1]}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # Separate features and target variables
    # Assuming 'Attack_label' and 'Attack_type' are the target columns
    target_cols = ['Attack_label', 'Attack_type']
    
    # Check if target columns exist
    available_targets = [col for col in target_cols if col in df.columns]
    if not available_targets:
        print("Warning: Target columns not found. Using last column as target.")
        target_col = df.columns[-1]
    else:
        # Use Attack_label for binary classification (prioritize if both exist)
        target_col = 'Attack_label' if 'Attack_label' in available_targets else available_targets[0]
    
    print(f"Using '{target_col}' as target variable")
    
    # Separate features and target
    X = df.drop([col for col in target_cols if col in df.columns], axis=1)
    y = df[target_col]
    
    # Handle non-numeric features if any
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Encode target variable if it's categorical
    label_encoder = None
    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:")
    target_counts = pd.Series(y).value_counts()
    print(target_counts)
    
    # Check for class imbalance
    imbalance_ratio = target_counts.min() / target_counts.max()
    print(f"Class imbalance ratio: {imbalance_ratio:.3f}")
    if imbalance_ratio < 0.1:
        print("⚠️  Severe class imbalance detected - this may lead to overfitting!")
    elif imbalance_ratio < 0.3:
        print("⚠️  Moderate class imbalance detected - monitoring for overfitting recommended")
    
    # Scale features for models that need it (SVM, KNN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split the data (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nData split completed:")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test, X.columns.tolist(), label_encoder, scaler



def enhanced_cross_validation(model, X_train, y_train, model_name, cv_folds=5):
    """
    Enhanced cross-validation with overfitting detection
    
    Args:
        model: The machine learning model
        X_train, y_train: Training data
        model_name: Name of the model for reporting
        cv_folds: Number of CV folds
    
    Returns:
        dict: Comprehensive CV results with overfitting indicators
    """
    print(f"\n🔍 Enhanced Cross-Validation for {model_name}")
    print("-" * 50)
    
    # Stratified K-Fold for better handling of imbalanced data
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy')
    
    # Additional metrics
    precision_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='precision_macro')
    recall_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='recall_macro')
    f1_scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='f1_macro')
    
    # Training vs Validation performance (overfitting check)
    train_scores = []
    val_scores = []
    
    for train_idx, val_idx in skf.split(X_train, y_train):
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        # Train on fold
        model.fit(X_train_fold, y_train_fold)
        
        # Score on training and validation
        train_score = model.score(X_train_fold, y_train_fold)
        val_score = model.score(X_val_fold, y_val_fold)
        
        train_scores.append(train_score)
        val_scores.append(val_score)
    
    train_scores = np.array(train_scores)
    val_scores = np.array(val_scores)
    
    # Overfitting indicators
    mean_train_score = train_scores.mean()
    mean_val_score = val_scores.mean()
    overfitting_gap = mean_train_score - mean_val_score
    
    # Variance analysis
    cv_variance = cv_scores.var()
    
    print(f"Cross-validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Cross-validation Precision: {precision_scores.mean():.4f} (+/- {precision_scores.std() * 2:.4f})")
    print(f"Cross-validation Recall: {recall_scores.mean():.4f} (+/- {recall_scores.std() * 2:.4f})")
    print(f"Cross-validation F1-Score: {f1_scores.mean():.4f} (+/- {f1_scores.std() * 2:.4f})")
    
    print(f"\n📊 Overfitting Analysis:")
    print(f"Training Score: {mean_train_score:.4f}")
    print(f"Validation Score: {mean_val_score:.4f}")
    print(f"Overfitting Gap: {overfitting_gap:.4f}")
    print(f"CV Variance: {cv_variance:.6f}")
    
    # Overfitting warnings
    if overfitting_gap > 0.05:
        print("🚨 WARNING: Significant overfitting detected!")
        if overfitting_gap > 0.1:
            print("🚨 CRITICAL: Severe overfitting - model may not generalize well!")
    elif overfitting_gap > 0.02:
        print("⚠️  Mild overfitting detected - monitor performance")
    else:
        print("✅ No significant overfitting detected")
    
    if cv_variance > 0.01:
        print("⚠️  High variance in CV scores - model may be unstable")
    
    return {
        'cv_accuracy_mean': cv_scores.mean(),
        'cv_accuracy_std': cv_scores.std(),
        'cv_precision_mean': precision_scores.mean(),
        'cv_recall_mean': recall_scores.mean(),
        'cv_f1_mean': f1_scores.mean(),
        'train_score_mean': mean_train_score,
        'val_score_mean': mean_val_score,
        'overfitting_gap': overfitting_gap,
        'cv_variance': cv_variance,
        'individual_cv_scores': cv_scores
    }

def plot_learning_curves(model, X_train, y_train, model_name):
    """
    Plot learning curves to visualize overfitting
    """
    print(f"Generating learning curves for {model_name}...")
    
    # Generate learning curve data
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        random_state=42
    )
    
    # Calculate means and standard deviations
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    # Plot learning curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy Score')
    plt.title(f'Learning Curves - {model_name}')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(f'learning_curve_{model_name.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return train_sizes, train_scores, val_scores

def train_decision_tree(X_train, X_test, y_train, y_test):
    """
    Train and evaluate Decision Tree model with enhanced overfitting detection
    """
    print("\n" + "="*50)
    print("Training Decision Tree Classifier")
    print("="*50)
    
    start_time = time.time()
    
    # Initialize the model with regularization to prevent overfitting
    dt_model = DecisionTreeClassifier(
        random_state=42,
        max_depth=10,          # Limit depth to prevent overfitting
        min_samples_split=5,   # Minimum samples to split
        min_samples_leaf=2,    # Minimum samples in leaf
        criterion='gini',
        min_impurity_decrease=0.001,  # Additional regularization
        ccp_alpha=0.01        # Cost complexity pruning
    )
    
    # Enhanced cross-validation
    cv_results = enhanced_cross_validation(dt_model, X_train, y_train, "Decision Tree")
    
    # Train the model
    dt_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    start_pred = time.time()
    y_pred = dt_model.predict(X_test)
    prediction_time = time.time() - start_pred
    
    # Calculate test accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Final overfitting check
    train_accuracy = dt_model.score(X_train, y_train)
    final_overfitting_gap = train_accuracy - test_accuracy
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Prediction completed in {prediction_time:.2f} seconds")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Final Overfitting Gap: {final_overfitting_gap:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Generate learning curves
    try:
        plot_learning_curves(dt_model, X_train, y_train, "Decision Tree")
    except Exception as e:
        print(f"Learning curve generation failed: {e}")
    
    return {
        'model': dt_model,
        'model_name': 'Decision Tree',
        'test_accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'final_overfitting_gap': final_overfitting_gap,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'y_pred': y_pred,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'cv_results': cv_results
    }

def train_random_forest(X_train, X_test, y_train, y_test):
    """
    Train and evaluate Random Forest model with enhanced overfitting detection
    """
    print("\n" + "="*50)
    print("Training Random Forest Classifier")
    print("="*50)
    
    start_time = time.time()
    
    # Initialize the model with regularization
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=15,          # Limit depth
        min_samples_split=5,   # Minimum samples to split
        min_samples_leaf=2,    # Minimum samples in leaf
        max_features='sqrt',   # Feature subsampling
        bootstrap=True,        # Bootstrap sampling
        oob_score=True,        # Out-of-bag score for validation
        n_jobs=-1
    )
    
    # Enhanced cross-validation
    cv_results = enhanced_cross_validation(rf_model, X_train, y_train, "Random Forest")
    
    # Train the model
    rf_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    start_pred = time.time()
    y_pred = rf_model.predict(X_test)
    prediction_time = time.time() - start_pred
    
    # Calculate test accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Final overfitting check
    train_accuracy = rf_model.score(X_train, y_train)
    final_overfitting_gap = train_accuracy - test_accuracy
    
    # Out-of-bag score (another validation metric)
    oob_score = rf_model.oob_score_
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Prediction completed in {prediction_time:.2f} seconds")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Out-of-Bag Score: {oob_score:.4f}")
    print(f"Final Overfitting Gap: {final_overfitting_gap:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Generate learning curves
    try:
        plot_learning_curves(rf_model, X_train, y_train, "Random Forest")
    except Exception as e:
        print(f"Learning curve generation failed: {e}")
    
    return {
        'model': rf_model,
        'model_name': 'Random Forest',
        'test_accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'oob_score': oob_score,
        'final_overfitting_gap': final_overfitting_gap,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'y_pred': y_pred,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'feature_importance': rf_model.feature_importances_,
        'cv_results': cv_results
    }

def train_knn(X_train, X_test, y_train, y_test):
    """
    Train and evaluate KNN model with enhanced overfitting detection
    """
    print("\n" + "="*50)
    print("Training K-Nearest Neighbors Classifier")
    print("="*50)
    
    start_time = time.time()
    
    # Test different k values with cross-validation to prevent overfitting
    k_values = [3, 5, 7, 9, 11, 15, 21]
    best_k = 5
    best_cv_score = 0
    k_cv_results = {}
    
    print("Testing different k values with cross-validation...")
    for k in k_values:
        knn_temp = KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_val_score(knn_temp, X_train, y_train, cv=5)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        k_cv_results[k] = {'mean': cv_mean, 'std': cv_std}
        print(f"k={k}: CV Score = {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
        
        if cv_mean > best_cv_score:
            best_cv_score = cv_mean
            best_k = k
    
    print(f"Best k value: {best_k}")
    
    # Initialize the model with best k
    knn_model = KNeighborsClassifier(n_neighbors=best_k)
    
    # Enhanced cross-validation
    cv_results = enhanced_cross_validation(knn_model, X_train, y_train, "K-Nearest Neighbors")
    
    # Train the model
    knn_model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Make predictions
    start_pred = time.time()
    y_pred = knn_model.predict(X_test)
    prediction_time = time.time() - start_pred
    
    # Calculate test accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Final overfitting check
    train_accuracy = knn_model.score(X_train, y_train)
    final_overfitting_gap = train_accuracy - test_accuracy
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Prediction completed in {prediction_time:.2f} seconds")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Final Overfitting Gap: {final_overfitting_gap:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Generate learning curves
    try:
        plot_learning_curves(knn_model, X_train, y_train, "K-Nearest Neighbors")
    except Exception as e:
        print(f"Learning curve generation failed: {e}")
    
    return {
        'model': knn_model,
        'model_name': 'K-Nearest Neighbors',
        'test_accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'final_overfitting_gap': final_overfitting_gap,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'y_pred': y_pred,
        'best_k': best_k,
        'k_cv_results': k_cv_results,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'cv_results': cv_results
    }

def train_svm(X_train, X_test, y_train, y_test):
    """
    Train and evaluate SVM model with enhanced overfitting detection
    """
    print("\n" + "="*50)
    print("Training Support Vector Machine")
    print("="*50)
    
    # For large datasets, use a subset for SVM to avoid memory issues
    if len(X_train) > 160000:
        print("Large dataset detected. Using subset for SVM training...")
        subset_size = 5000
        indices = np.random.choice(len(X_train), subset_size, replace=False)
        X_train_svm = X_train[indices]
        y_train_svm = y_train[indices]
    else:
        X_train_svm = X_train
        y_train_svm = y_train
    
    start_time = time.time()
    
    # Test different configurations with cross-validation
    configs = [
        {'kernel': 'linear', 'C': 1.0},
        {'kernel': 'rbf', 'C': 1.0},
        {'kernel': 'rbf', 'C': 0.1},  # Regularized version
        {'kernel': 'rbf', 'C': 10.0}  # Less regularized
    ]
    
    best_config = {'kernel': 'rbf', 'C': 1.0}
    best_cv_score = 0
    config_results = {}
    
    print("Testing different SVM configurations...")
    for config in configs:
        svm_temp = SVC(**config, random_state=42)
        cv_scores = cross_val_score(svm_temp, X_train_svm, y_train_svm, cv=3)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        config_key = f"{config['kernel']}_C{config['C']}"
        config_results[config_key] = {'mean': cv_mean, 'std': cv_std}
        print(f"{config_key}: CV Score = {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
        
        if cv_mean > best_cv_score:
            best_cv_score = cv_mean
            best_config = config
    
    print(f"Best configuration: {best_config}")
    
    # Initialize the model with best configuration
    svm_model = SVC(**best_config, random_state=42)
    
    # Enhanced cross-validation
    cv_results = enhanced_cross_validation(svm_model, X_train_svm, y_train_svm, "Support Vector Machine")
    
    # Train the model
    svm_model.fit(X_train_svm, y_train_svm)
    training_time = time.time() - start_time
    
    # Make predictions
    start_pred = time.time()
    y_pred = svm_model.predict(X_test)
    prediction_time = time.time() - start_pred
    
    # Calculate test accuracy
    test_accuracy = accuracy_score(y_test, y_pred)
    
    # Final overfitting check (on subset used for training)
    train_accuracy = svm_model.score(X_train_svm, y_train_svm)
    final_overfitting_gap = train_accuracy - test_accuracy
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Prediction completed in {prediction_time:.2f} seconds")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Training Accuracy (subset): {train_accuracy:.4f}")
    print(f"Final Overfitting Gap: {final_overfitting_gap:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Generate learning curves (on subset)
    try:
        plot_learning_curves(svm_model, X_train_svm, y_train_svm, "Support Vector Machine")
    except Exception as e:
        print(f"Learning curve generation failed: {e}")
    
    return {
        'model': svm_model,
        'model_name': 'Support Vector Machine',
        'test_accuracy': test_accuracy,
        'train_accuracy': train_accuracy,
        'final_overfitting_gap': final_overfitting_gap,
        'training_time': training_time,
        'prediction_time': prediction_time,
        'y_pred': y_pred,
        'best_config': best_config,
        'config_results': config_results,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'cv_results': cv_results
    }

def plot_confusion_matrices(models_results, y_test):
    """
    Plot confusion matrices for all models
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, result in enumerate(models_results):
        cm = confusion_matrix(y_test, result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues')
        axes[i].set_title(f'{result["model_name"]} - Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_overfitting_analysis(models_results):
    """
    Plot overfitting analysis for all models
    """
    model_names = [result['model_name'] for result in models_results]
    train_scores = [result['train_accuracy'] for result in models_results]
    test_scores = [result['test_accuracy'] for result in models_results]
    overfitting_gaps = [result['final_overfitting_gap'] for result in models_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training vs Test Accuracy
    x = np.arange(len(model_names))
    width = 0.35
    
    ax1.bar(x - width/2, train_scores, width, label='Training Accuracy', alpha=0.8)
    ax1.bar(x + width/2, test_scores, width, label='Test Accuracy', alpha=0.8)
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training vs Test Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Overfitting Gap
    colors = ['red' if gap > 0.05 else 'orange' if gap > 0.02 else 'green' for gap in overfitting_gaps]
    ax2.bar(x, overfitting_gaps, color=colors, alpha=0.8)
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Overfitting Gap')
    ax2.set_title('Overfitting Analysis')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Significant Overfitting')
    ax2.axhline(y=0.02, color='orange', linestyle='--', alpha=0.7, label='Mild Overfitting')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('overfitting_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance(rf_result, feature_names, top_n=20):
    """
    Plot feature importance for Random Forest
    """
    if 'feature_importance' in rf_result:
        # Get top N features
        importance = rf_result['feature_importance']
        indices = np.argsort(importance)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top {top_n} Feature Importance - Random Forest')
        plt.bar(range(top_n), importance[indices])
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

def compare_models(models_results):
    """
    Compare performance of all models with enhanced overfitting metrics
    
    Args:
        models_results (list): List of model results
    """
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON WITH OVERFITTING ANALYSIS")
    print("="*80)
    
    # Create comparison dataframe
    comparison_data = []
    for result in models_results:
        cv_results = result['cv_results']
        comparison_data.append({
            'Model': result['model_name'],
            'Test Accuracy': result['test_accuracy'],
            'Train Accuracy': result['train_accuracy'],
            'Overfitting Gap': result['final_overfitting_gap'],
            'CV Mean': cv_results['cv_accuracy_mean'],
            'CV Std': cv_results['cv_accuracy_std'],
            'CV Variance': cv_results['cv_variance'],
            'Training Time (s)': result['training_time'],
            'Prediction Time (s)': result['prediction_time']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    
    # Find best model based on test accuracy and overfitting
    print("\n" + "="*80)
    print("MODEL RANKING ANALYSIS")
    print("="*80)
    
    # Best test accuracy
    best_accuracy_idx = comparison_df['Test Accuracy'].idxmax()
    best_accuracy_model = comparison_df.iloc[best_accuracy_idx]
    
    # Least overfitting
    least_overfitting_idx = comparison_df['Overfitting Gap'].idxmin()
    least_overfitting_model = comparison_df.iloc[least_overfitting_idx]
    
    # Most stable (lowest CV variance)
    most_stable_idx = comparison_df['CV Variance'].idxmin()
    most_stable_model = comparison_df.iloc[most_stable_idx]
    
    print(f"🎯 HIGHEST TEST ACCURACY: {best_accuracy_model['Model']}")
    print(f"   Test Accuracy: {best_accuracy_model['Test Accuracy']:.4f}")
    print(f"   Overfitting Gap: {best_accuracy_model['Overfitting Gap']:.4f}")
    
    print(f"\n✅ LEAST OVERFITTING: {least_overfitting_model['Model']}")
    print(f"   Overfitting Gap: {least_overfitting_model['Overfitting Gap']:.4f}")
    print(f"   Test Accuracy: {least_overfitting_model['Test Accuracy']:.4f}")
    
    print(f"\n📊 MOST STABLE: {most_stable_model['Model']}")
    print(f"   CV Variance: {most_stable_model['CV Variance']:.6f}")
    print(f"   Test Accuracy: {most_stable_model['Test Accuracy']:.4f}")
    
    # Overall recommendation
    print(f"\n🏆 RECOMMENDED MODEL:")
    # Score models based on test accuracy and inverse overfitting gap
    comparison_df['Combined Score'] = (
        comparison_df['Test Accuracy'] - 
        comparison_df['Overfitting Gap'] * 0.5 -  # Penalty for overfitting
        comparison_df['CV Variance'] * 10  # Penalty for instability
    )
    
    best_overall_idx = comparison_df['Combined Score'].idxmax()
    best_overall_model = comparison_df.iloc[best_overall_idx]
    
    print(f"   {best_overall_model['Model']}")
    print(f"   Test Accuracy: {best_overall_model['Test Accuracy']:.4f}")
    print(f"   Overfitting Gap: {best_overall_model['Overfitting Gap']:.4f}")
    print(f"   CV Variance: {best_overall_model['CV Variance']:.6f}")
    
    # Overfitting warnings
    print("\n" + "="*80)
    print("OVERFITTING WARNINGS")
    print("="*80)
    
    for _, row in comparison_df.iterrows():
        gap = row['Overfitting Gap']
        variance = row['CV Variance']
        
        if gap > 0.1:
            print(f"🚨 CRITICAL: {row['Model']} - Severe overfitting (gap: {gap:.4f})")
        elif gap > 0.05:
            print(f"⚠️  WARNING: {row['Model']} - Significant overfitting (gap: {gap:.4f})")
        elif gap > 0.02:
            print(f"⚠️  CAUTION: {row['Model']} - Mild overfitting (gap: {gap:.4f})")
        else:
            print(f"✅ GOOD: {row['Model']} - No significant overfitting (gap: {gap:.4f})")
        
        if variance > 0.01:
            print(f"   ⚠️  High variance detected - model may be unstable")
    
    return comparison_df

def save_best_model(models_results, comparison_df):
    """
    Save the best performing model (considering overfitting)
    """
    # Use the model with best combined score (accuracy - overfitting penalty)
    best_model_idx = comparison_df['Combined Score'].idxmax()
    best_model_result = models_results[best_model_idx]
    best_model = best_model_result['model']
    model_name = best_model_result['model_name'].replace(' ', '_').lower()
    
    filename = f'best_model_{model_name}.joblib'
    joblib.dump(best_model, filename)
    print(f"\n💾 Best model saved as: {filename}")
    print(f"   Model: {best_model_result['model_name']}")
    print(f"   Test Accuracy: {best_model_result['test_accuracy']:.4f}")
    print(f"   Overfitting Gap: {best_model_result['final_overfitting_gap']:.4f}")
    
    return filename

def generate_overfitting_recommendations(models_results, comparison_df):
    """
    Generate specific recommendations to address overfitting
    """
    print("\n" + "="*80)
    print("OVERFITTING PREVENTION RECOMMENDATIONS")
    print("="*80)
    
    # General recommendations
    print("🔧 GENERAL RECOMMENDATIONS:")
    print("   • Use cross-validation for all model selections")
    print("   • Monitor training vs validation performance")
    print("   • Implement early stopping when possible")
    print("   • Use regularization techniques")
    print("   • Collect more training data if possible")
    
    # Model-specific recommendations
    print("\n🎯 MODEL-SPECIFIC RECOMMENDATIONS:")
    
    for result in models_results:
        model_name = result['model_name']
        gap = result['final_overfitting_gap']
        
        print(f"\n{model_name}:")
        if gap > 0.05:
            if 'Decision Tree' in model_name:
                print("   • Increase min_samples_split and min_samples_leaf")
                print("   • Reduce max_depth")
                print("   • Use cost complexity pruning (ccp_alpha)")
            elif 'Random Forest' in model_name:
                print("   • Reduce max_depth")
                print("   • Increase min_samples_split")
                print("   • Use fewer features (max_features)")
            elif 'KNN' in model_name:
                print("   • Increase k value")
                print("   • Use distance weighting")
                print("   • Apply feature selection")
            elif 'SVM' in model_name:
                print("   • Reduce C parameter (increase regularization)")
                print("   • Use simpler kernel (linear instead of RBF)")
                print("   • Apply feature scaling")
        else:
            print("   ✅ Good generalization - no specific changes needed")
    
    # Data recommendations
    print("\n📊 DATA RECOMMENDATIONS:")
    high_overfitting_models = [r for r in models_results if r['final_overfitting_gap'] > 0.05]
    
    if high_overfitting_models:
        print("   • Consider data augmentation techniques")
        print("   • Implement feature selection to reduce dimensionality")
        print("   • Check for data leakage")
        print("   • Ensure proper train/validation/test split")
        print("   • Balance classes if severe imbalance exists")
    else:
        print("   ✅ Current data preprocessing appears adequate")
    
    # Deployment recommendations
    print("\n🚀 DEPLOYMENT RECOMMENDATIONS:")
    best_model_name = comparison_df.loc[comparison_df['Combined Score'].idxmax(), 'Model']
    print(f"   • Deploy {best_model_name} for production")
    print("   • Implement monitoring for performance drift")
    print("   • Set up alerts for accuracy degradation")
    print("   • Plan for model retraining schedule")
    print("   • Use A/B testing for model updates")

def main():
    """
    Main execution function with enhanced overfitting detection
    """
    print("🔒 EdgeIIoTset Network Security Analysis")
    print("🤖 Machine Learning Models with Overfitting Detection")
    print("="*80)
    
    # File path - update this to your actual dataset path
    file_path = "Preprocessed DataSet/Preprocessed-ML-EdgeIIoT-dataset.csv"  # Update this path
    
    # Load and prepare data
    X_train, X_test, y_train, y_test, feature_names, label_encoder, scaler = load_and_prepare_data(file_path)
    
    # Convert to numpy arrays if they're pandas DataFrames
    if hasattr(X_train, 'values'):
        X_train = X_train.values
        X_test = X_test.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values
        y_test = y_test.values
    
    # Train all models with enhanced overfitting detection
    models_results = []
    
    # Decision Tree
    print("\n🌳 Training Decision Tree with overfitting detection...")
    dt_result = train_decision_tree(X_train, X_test, y_train, y_test)
    models_results.append(dt_result)
    
    # Random Forest
    print("\n🌲 Training Random Forest with overfitting detection...")
    rf_result = train_random_forest(X_train, X_test, y_train, y_test)
    models_results.append(rf_result)
    
    # K-Nearest Neighbors
    print("\n🎯 Training K-Nearest Neighbors with overfitting detection...")
    knn_result = train_knn(X_train, X_test, y_train, y_test)
    models_results.append(knn_result)
    
    # Support Vector Machine
    print("\n⚡ Training Support Vector Machine with overfitting detection...")
    svm_result = train_svm(X_train, X_test, y_train, y_test)
    models_results.append(svm_result)
    
    # Compare models with overfitting analysis
    comparison_df = compare_models(models_results)
    
    # Generate visualizations
    try:
        print("\n📊 Generating visualizations...")
        plot_confusion_matrices(models_results, y_test)
        plot_overfitting_analysis(models_results)
        plot_feature_importance(rf_result, feature_names)
    except Exception as e:
        print(f"Visualization error: {e}")
    
    # Save best model (considering overfitting)
    try:
        save_best_model(models_results, comparison_df)
    except Exception as e:
        print(f"Model saving error: {e}")
    
    # Generate overfitting-specific recommendations
    generate_overfitting_recommendations(models_results, comparison_df)
    
    print("\n" + "="*80)
    print("✅ ENHANCED ANALYSIS COMPLETE")
    print("="*80)
    print("All models have been trained with comprehensive overfitting detection!")
    print("Key improvements implemented:")
    print("• Enhanced cross-validation with overfitting detection")
    print("• Learning curves for visual overfitting analysis")
    print("• Regularization parameters for all models")
    print("• Comprehensive overfitting gap analysis")
    print("• Model stability assessment via CV variance")
    print("• Specific recommendations for overfitting prevention")
    print("\nCheck the generated files for detailed results and visualizations.")

if __name__ == "__main__":
    main()