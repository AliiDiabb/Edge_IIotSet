import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                           confusion_matrix, classification_report, roc_auc_score)
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, InputLayer
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import to_categorical
import warnings
import os
warnings.filterwarnings('ignore')

print("Checking GPU availability...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for GPU to prevent allocation errors
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU(s) available: {len(gpus)}")
        print(f"GPU Details: {[gpu.name for gpu in gpus]}")
        
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU found. Using CPU.")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------
# FLOPS Calculation Function
# ---------------------------
def get_flops(model, batch_size=1):
    """
    Calculate FLOPs for a given model
    """
    flops = 0
    
    for layer in model.layers:
        if isinstance(layer, Dense):
            # For Dense layers: FLOPs = 2 * input_size * output_size
            input_size = layer.input_shape[-1]
            output_size = layer.units
            flops += 2 * input_size * output_size
        # Add more layer types as needed
    
    return flops * batch_size

# ---------------------------
# Hardware Measures Function
# ---------------------------
def hw_measures(model):
    n_params = model.count_params()
    max_tens = max([np.prod(layer.output_shape[1:]) for layer in model.layers if None not in layer.output_shape[1:]], default=0)
    flops = get_flops(model, batch_size=1)
    flash_size = 4 * n_params  # 4 bytes per parameter
    ram_size = 4 * max_tens    # 4 bytes per tensor
    return n_params, max_tens, flops, flash_size, ram_size

# ---------------------------
# Fixed Edge-IIoT Shallow DNN Model
# ---------------------------
def build_edgeiiot_dnn(input_shape=(108,), num_classes=2, classification_type='binary'):
    """
    Build a very shallow DNN model with minimal parameters for Edge-IIoT
    Fixed to handle binary classification correctly
    """
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(Dense(2, activation='relu'))  # Keep units very small to match 47 parameters
    
    # FIX: Use different output layers for binary vs multi-class
    if classification_type == 'binary':
        # Binary classification: single output unit with sigmoid
        model.add(Dense(1, activation='sigmoid'))
    else:
        # Multi-class classification: multiple output units with softmax
        model.add(Dense(num_classes, activation='softmax'))
    
    return model

class DNNIDSAnalyzer:
    def __init__(self, dataset_path):
        """
        Initialize the DNN IDS Analyzer
        
        Args:
            dataset_path (str): Path to the preprocessed Edge-IIoT dataset
        """
        self.dataset_path = dataset_path
        self.data = None
        self.X = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.results = {}
        
    def load_and_prepare_data(self, max_samples=2218386):
        """
        Load and prepare the dataset for analysis
        
        Args:
            max_samples (int): Maximum number of samples to use (2,219,000)
        """
        print("Loading Edge-IIoT dataset...")
        
        # Load dataset with efficient memory usage
        try:
            self.data = pd.read_csv(self.dataset_path, low_memory=False)
            print(f"Dataset loaded successfully: {self.data.shape}")
        except FileNotFoundError:
            print(f"ERROR: Dataset file not found: {self.dataset_path}")
            print("Please ensure the dataset file exists at the specified path.")
            return False
        except Exception as e:
            print(f"ERROR loading dataset: {e}")
            return False
        
        # Use maximum specified samples
        if len(self.data) > max_samples:
            self.data = self.data.sample(n=max_samples, random_state=42)
            print(f"Using {max_samples:,} samples from the dataset")
        
        # Basic data exploration
        print("\n=== Dataset Overview ===")
        print(f"Shape: {self.data.shape}")
        print(f"Features: {self.data.shape[1]}")
        print(f"Samples: {self.data.shape[0]:,}")
        print(f"Columns: {list(self.data.columns)}")
        
        # Check for target columns
        target_columns = [col for col in self.data.columns if 'attack' in col.lower() or 'label' in col.lower()]
        print(f"Potential target columns found: {target_columns}")
        
        if not target_columns:
            print("ERROR: No target columns found! Please check your dataset.")
            print("Expected columns with 'attack' or 'label' in the name.")
            return False
        
        # Use the first target column found
        target_col = target_columns[0]
        print(f"Using '{target_col}' as target column")
        
        # Prepare features (exclude target columns)
        feature_cols = [col for col in self.data.columns if col not in target_columns]
        self.X = self.data[feature_cols].copy()
        
        # Handle non-numeric features
        for col in self.X.columns:
            if self.X[col].dtype == 'object':
                le = LabelEncoder()
                self.X[col] = le.fit_transform(self.X[col].astype(str))
        
        print(f"Final feature matrix shape: {self.X.shape}")
        
        # Store target column name for later use
        self.target_column = target_col
        
        # Data split: 80% train, 10% validation, 10% test
        print("\n=== Splitting Data (80%-10%-10%) ===")
        
        try:
            # First split: 90% temp, 10% test
            X_temp, X_test, y_temp, y_test = train_test_split(
                self.X, self.data[target_col], test_size=0.1, random_state=42, 
                stratify=self.data[target_col] if len(self.data[target_col].unique()) > 1 else None
            )
            
            # Second split: 80% train, 10% validation from temp
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=0.111, random_state=42,  # 0.111 * 0.9 ≈ 0.1
                stratify=y_temp if len(y_temp.unique()) > 1 else None
            )
            
            self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
            self.y_train, self.y_val, self.y_test = y_train, y_val, y_test
            
        except Exception as e:
            print(f"Error during data splitting: {e}")
            return False
        
        # Feature scaling
        print("Applying feature scaling...")
        try:
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_val_scaled = self.scaler.transform(self.X_val)
            self.X_test_scaled = self.scaler.transform(self.X_test)
        except Exception as e:
            print(f"Error during feature scaling: {e}")
            return False
        
        print(f"Training set: {self.X_train_scaled.shape} ({len(self.X_train_scaled)/len(self.X)*100:.1f}%)")
        print(f"Validation set: {self.X_val_scaled.shape} ({len(self.X_val_scaled)/len(self.X)*100:.1f}%)")
        print(f"Test set: {self.X_test_scaled.shape} ({len(self.X_test_scaled)/len(self.X)*100:.1f}%)")
        
        return True
    
    def prepare_targets_for_classification(self, classification_type):
        """
        Prepare target variables based on classification type
        
        Args:
            classification_type (str): 'binary', '6class', or '15class'
        """
        print(f"\n=== Preparing targets for {classification_type} classification ===")
        
        try:
            # Reset scaled data to original state for each classification type
            X_train_scaled = self.X_train_scaled.copy()
            X_val_scaled = self.X_val_scaled.copy()
            X_test_scaled = self.X_test_scaled.copy()
            y_train = self.y_train.copy()
            y_val = self.y_val.copy()
            y_test = self.y_test.copy()
            
            if classification_type == 'binary':
                # Binary: Normal vs Attack - assuming 'Normal' or 0 is normal traffic
                unique_labels = y_train.unique()
                print(f"Unique labels in training data: {unique_labels}")
                
                # Try to identify normal vs attack
                if 'Normal' in unique_labels:
                    y_train = (y_train != 'Normal').astype(int)
                    y_val = (y_val != 'Normal').astype(int)
                    y_test = (y_test != 'Normal').astype(int)
                elif 0 in unique_labels:
                    y_train = (y_train != 0).astype(int)
                    y_val = (y_val != 0).astype(int)
                    y_test = (y_test != 0).astype(int)
                else:
                    # Assume the most frequent label is normal
                    normal_label = y_train.value_counts().index[0]
                    print(f"Assuming '{normal_label}' is the normal class")
                    y_train = (y_train != normal_label).astype(int)
                    y_val = (y_val != normal_label).astype(int)
                    y_test = (y_test != normal_label).astype(int)
                
                # FIX: Ensure binary targets are properly shaped (1D arrays)
                y_train = y_train.values.reshape(-1) if hasattr(y_train, 'values') else y_train.reshape(-1)
                y_val = y_val.values.reshape(-1) if hasattr(y_val, 'values') else y_val.reshape(-1)
                y_test = y_test.values.reshape(-1) if hasattr(y_test, 'values') else y_test.reshape(-1)
                
                num_classes = 2
                
            elif classification_type == '6class':
                # 6 classes: Use LabelEncoder and limit to 6 most frequent classes
                all_labels = pd.concat([y_train, y_val, y_test])
                
                # Get top 6 most frequent classes
                top_classes = all_labels.value_counts().head(6).index.tolist()
                print(f"Top 6 classes: {top_classes}")
                
                # Filter data to only include top 6 classes
                train_mask = y_train.isin(top_classes)
                val_mask = y_val.isin(top_classes)
                test_mask = y_test.isin(top_classes)
                
                # Apply masks consistently
                X_train_scaled = X_train_scaled[train_mask]
                X_val_scaled = X_val_scaled[val_mask]
                X_test_scaled = X_test_scaled[test_mask]
                
                y_train_filtered = y_train[train_mask]
                y_val_filtered = y_val[val_mask]
                y_test_filtered = y_test[test_mask]
                
                # Encode labels
                le = LabelEncoder()
                le.fit(top_classes)
                y_train = le.transform(y_train_filtered)
                y_val = le.transform(y_val_filtered)
                y_test = le.transform(y_test_filtered)
                
                num_classes = 6
                print(f"6-class labels: {le.classes_}")
                
            else:  # 15class
                # 15 classes: Use LabelEncoder on all classes, limit to top 15
                all_labels = pd.concat([y_train, y_val, y_test])
                
                # Get top 15 most frequent classes
                top_classes = all_labels.value_counts().head(15).index.tolist()
                print(f"Top 15 classes: {top_classes}")
                
                # Filter data to only include top 15 classes
                train_mask = y_train.isin(top_classes)
                val_mask = y_val.isin(top_classes)
                test_mask = y_test.isin(top_classes)
                
                # Apply masks consistently
                X_train_scaled = X_train_scaled[train_mask]
                X_val_scaled = X_val_scaled[val_mask]
                X_test_scaled = X_test_scaled[test_mask]
                
                y_train_filtered = y_train[train_mask]
                y_val_filtered = y_val[val_mask]
                y_test_filtered = y_test[test_mask]
                
                # Encode labels
                le = LabelEncoder()
                le.fit(top_classes)
                y_train = le.transform(y_train_filtered)
                y_val = le.transform(y_val_filtered)
                y_test = le.transform(y_test_filtered)
                
                num_classes = len(le.classes_)
                print(f"15-class labels: {le.classes_}")
                
                # Store label encoder for later reference
                self.label_encoder_15class = le
            
            # Print class distribution
            unique, counts = np.unique(y_train, return_counts=True)
            print(f"Training class distribution: {dict(zip(unique, counts))}")
            print(f"Final data shapes - Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")
            print(f"Target shapes - Train: {y_train.shape}, Val: {y_val.shape}, Test: {y_test.shape}")
            
            return y_train, y_val, y_test, num_classes, X_train_scaled, X_val_scaled, X_test_scaled
            
        except Exception as e:
            print(f"Error preparing targets for {classification_type}: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None, None, None, None, None
    
    def build_dnn_model(self, num_classes, classification_type, input_dim):
        """
        Build and compile the Shallow DNN model for Edge-IIoT
        Fixed to handle binary classification correctly
        
        Args:
            num_classes (int): Number of output classes
            classification_type (str): Type of classification for naming
            input_dim (int): Input dimension
        """
        print(f"\n=== Building Shallow DNN Model for {classification_type} ===")
        
        try:
            # Use the fixed shallow DNN architecture
            model = build_edgeiiot_dnn(input_shape=(input_dim,), num_classes=num_classes, classification_type=classification_type)
            
            # FIX: Use appropriate loss function for each case
            if classification_type == 'binary':
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='binary_crossentropy',  # Binary classification with sigmoid output
                    metrics=['accuracy']
                )
            else:
                model.compile(
                    optimizer=Adam(learning_rate=0.001),
                    loss='sparse_categorical_crossentropy',  # Multi-class with softmax output
                    metrics=['accuracy']
                )
            
            print(f"Shallow DNN architecture for {classification_type}:")
            model.summary()
            
            # Calculate and display hardware measures
            print(f"\n=== Hardware Measures for {classification_type} ===")
            params, max_tens, flops, flash_size, ram_size = hw_measures(model)
            print(f"Parameters       : {params}")
            print(f"Max Tensor Size  : {max_tens}")
            print(f"FLOPs            : {flops}")
            print(f"Flash Size (B)   : {flash_size}")
            print(f"RAM Size (B)     : {ram_size}")
            
            return model
            
        except Exception as e:
            print(f"Error building model for {classification_type}: {e}")
            return None
    
    def train_dnn_model(self, model, X_train, y_train, X_val, y_val, classification_type, epochs=100, batch_size=256):
        """
        Train the DNN model with proper callbacks
        """
        print(f"\n=== Training Shallow DNN Model for {classification_type} ===")
        
        try:
            # Force CPU usage for problematic GPU memory allocation
            with tf.device('/CPU:0'):
                # Callbacks for training optimization
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                )
                
                lr_reducer = ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=7,
                    min_lr=0.00001,
                    verbose=1
                )
                
                # Print shapes for debugging
                print(f"Training data shapes:")
                print(f"  X_train: {X_train.shape}")
                print(f"  y_train: {y_train.shape}")
                print(f"  X_val: {X_val.shape}")
                print(f"  y_val: {y_val.shape}")
                
                # Train the model
                print(f"Training with {len(X_train):,} samples on CPU...")
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping, lr_reducer],
                    verbose=1
                )
            
            print(f"Shallow DNN training for {classification_type} completed!")
            return history
            
        except Exception as e:
            print(f"Error training model for {classification_type}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_model(self, model, X_test, y_test, classification_type):
        """
        Evaluate model and calculate comprehensive metrics
        Fixed to handle binary classification predictions correctly
        """
        print(f"\n=== Evaluating {classification_type} Model ===")
        
        try:
            # Make predictions
            y_pred_prob = model.predict(X_test)
            
            # FIX: Handle predictions correctly for binary vs multi-class
            if classification_type == 'binary':
                # Binary classification: sigmoid output, threshold at 0.5
                y_pred = (y_pred_prob > 0.5).astype(int).flatten()
                y_pred_prob_for_auc = y_pred_prob.flatten()
            else:
                # Multi-class: softmax output, take argmax
                y_pred = np.argmax(y_pred_prob, axis=1)
                y_pred_prob_for_auc = y_pred_prob
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_macro': f1_score(y_test, y_pred, average='macro'),
                'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
                'precision_macro': precision_score(y_test, y_pred, average='macro'),
                'recall_macro': recall_score(y_test, y_pred, average='macro'),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            # ROC-AUC for binary classification
            if classification_type == 'binary':
                try:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_pred_prob_for_auc)
                except Exception as e:
                    print(f"Warning: Could not calculate ROC-AUC: {e}")
            
            self.results[classification_type] = metrics
            
            # Display results
            print(f"\n{classification_type.upper()} CLASSIFICATION RESULTS:")
            print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
            print(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
            print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
            print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
            
            if 'roc_auc' in metrics:
                print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"Error evaluating model for {classification_type}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_complete_analysis(self):
        """
        Run complete analysis for all three classification types
        """
        print("="*80)
        print("SHALLOW DEEP NEURAL NETWORK INTRUSION DETECTION SYSTEM")
        print("EDGE-IIOT OPTIMIZED MULTI-CLASSIFICATION ANALYSIS")
        print("="*80)
        
        if not self.load_and_prepare_data():
            print("Failed to load and prepare data. Exiting.")
            return False
        
        classification_types = ['binary', '6class', '15class']
        
        for class_type in classification_types:
            print(f"\n{'='*60}")
            print(f"RUNNING {class_type.upper()} CLASSIFICATION")
            print(f"{'='*60}")
            
            try:
                # Prepare targets
                result = self.prepare_targets_for_classification(class_type)
                if result[0] is None:
                    print(f"Failed to prepare targets for {class_type}. Skipping.")
                    continue
                
                y_train, y_val, y_test, num_classes, X_train_scaled, X_val_scaled, X_test_scaled = result
                
                # Build model
                model = self.build_dnn_model(num_classes, class_type, X_train_scaled.shape[1])
                
                if model is None:
                    print(f"Failed to build model for {class_type}. Skipping.")
                    continue
                
                # Train model
                history = self.train_dnn_model(model, X_train_scaled, y_train, X_val_scaled, y_val, class_type, epochs=50)
                
                if history is None:
                    print(f"Failed to train model for {class_type}. Skipping.")
                    continue
                
                # Evaluate model
                metrics = self.evaluate_model(model, X_test_scaled, y_test, class_type)
                
                if metrics is None:
                    print(f"Failed to evaluate model for {class_type}. Skipping.")
                    continue
                
                # Store model and history for visualization
                setattr(self, f'model_{class_type}', model)
                setattr(self, f'history_{class_type}', history)
                
            except Exception as e:
                print(f"Error processing {class_type} classification: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return len(self.results) > 0
    
    def visualize_results(self, save_path="./visualizations/"):
        """
        Create comprehensive visualizations for all classification types and save them
        """
        print(f"\n=== Generating and Saving Visualizations ===")
        
        # Check if we have results to visualize
        if not self.results:
            print("No results available for visualization. Please run the analysis first.")
            return
        
        # Create directory if it doesn't exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            print(f"Created directory: {save_path}")
        
        # Set up the plotting
        plt.style.use('default')
        
        # Get available classification types
        available_types = list(self.results.keys())
        colors = ['skyblue', 'lightgreen', 'lightcoral'][:len(available_types)]
        
        try:
            # 1. Model Performance Comparison
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Accuracy comparison
            accuracies = [self.results[ct]['accuracy'] for ct in available_types]
            bars1 = axes[0, 0].bar(available_types, accuracies, color=colors, alpha=0.7)
            axes[0, 0].set_title('Accuracy Comparison Across Classification Types')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, acc in zip(bars1, accuracies):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{acc:.3f}', ha='center', va='bottom')
            
            # F1-Score comparison
            f1_scores = [self.results[ct]['f1_weighted'] for ct in available_types]
            bars2 = axes[0, 1].bar(available_types, f1_scores, color=colors, alpha=0.7)
            axes[0, 1].set_title('F1-Score (Weighted) Comparison')
            axes[0, 1].set_ylabel('F1-Score')
            axes[0, 1].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, f1 in zip(bars2, f1_scores):
                axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{f1:.3f}', ha='center', va='bottom')
            
            # Precision vs Recall
            precisions = [self.results[ct]['precision_macro'] for ct in available_types]
            recalls = [self.results[ct]['recall_macro'] for ct in available_types]
            
            scatter = axes[1, 0].scatter(precisions, recalls, s=150, c=colors, alpha=0.7)
            for i, ct in enumerate(available_types):
                axes[1, 0].annotate(ct, (precisions[i], recalls[i]), 
                                  xytext=(5, 5), textcoords='offset points', fontsize=10)
            axes[1, 0].set_xlabel('Precision (Macro)')
            axes[1, 0].set_ylabel('Recall (Macro)')
            axes[1, 0].set_title('Precision vs Recall')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Training history comparison
            axes[1, 1].set_title('Training History Comparison')
            for i, ct in enumerate(available_types):
                if hasattr(self, f'history_{ct}'):
                    history = getattr(self, f'history_{ct}').history
                    epochs = range(1, len(history['val_accuracy']) + 1)
                    axes[1, 1].plot(epochs, history['val_accuracy'], 
                                   label=f'{ct} Validation Accuracy', color=colors[i])
            
            axes[1, 1].set_xlabel('Epochs')
            axes[1, 1].set_ylabel('Validation Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the performance comparison plot
            performance_plot_path = os.path.join(save_path, 'shallow_dnn_performance_comparison.png')
            plt.savefig(performance_plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {performance_plot_path}")
            plt.show()
            
            # Continue with other visualizations...
            print(f"\nAll visualizations saved to: {save_path}")
            
        except Exception as e:
            print(f"Error creating visualizations: {e}")
    
    def generate_comprehensive_report(self):
        """
        Generate comprehensive analysis report for all classification types
        """
        print(f"\n{'='*80}")
        print("COMPREHENSIVE ANALYSIS REPORT")
        print("SHALLOW DEEP NEURAL NETWORK INTRUSION DETECTION SYSTEM")
        print("EDGE-IIOT OPTIMIZED")
        print(f"{'='*80}")
        
        if not hasattr(self, 'data') or self.data is None:
            print("No data available for report generation. Please run the analysis first.")
            return
        
        print(f"\n1. DATASET OVERVIEW")
        print(f"   - Total Samples: {len(self.data):,}")
        print(f"   - Features: {self.X.shape[1] if hasattr(self, 'X') else 'N/A'}")
        if hasattr(self, 'X_train'):
            print(f"   - Training Samples: {len(self.X_train):,} (80%)")
            print(f"   - Validation Samples: {len(self.X_val):,} (10%)")
            print(f"   - Test Samples: {len(self.X_test):,} (10%)")
        
        if not self.results:
            print(f"\n2. RESULTS")
            print("   No results available. Analysis may have failed.")
            return
            
        print(f"\n2. PERFORMANCE SUMMARY")
        print(f"   {'Classification Type':<20} {'Accuracy':<12} {'F1-Macro':<12} {'F1-Weighted':<12}")
        print(f"   {'-'*60}")
        
        for ct in self.results:
            acc = self.results[ct]['accuracy']
            f1_macro = self.results[ct]['f1_macro']
            f1_weighted = self.results[ct]['f1_weighted']
            print(f"   {ct.upper():<20} {acc:.4f} ({acc*100:.1f}%)  {f1_macro:.4f}      {f1_weighted:.4f}")


# Example usage and main execution
if __name__ == "__main__":
    # Initialize the analyzer
    # Replace with your actual dataset path
    dataset_path = "Preprocessed DataSet/Preprocessed-DNN-EdgeIIoT-dataset.csv"  # Update this path
    
    analyzer = DNNIDSAnalyzer(dataset_path)
    
    # Run complete analysis
    success = analyzer.run_complete_analysis()
    
    if success:
        # Generate visualizations only if analysis was successful
        analyzer.visualize_results()
        
        # Generate comprehensive report
        analyzer.generate_comprehensive_report()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("Check the './visualizations/' folder for saved plots.")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("ANALYSIS FAILED!")
        print("Please check your dataset path and format.")
        print("="*80)