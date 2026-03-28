import sys
import os
import shutil
from datetime import datetime
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight

# Add the project root to sys.path to allow importing config.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from config import XGBOOST_PARAMS, PIPELINE

def train_model(verbose=False):
    # 1. Load Data
    data_dir = os.path.join(PROJECT_ROOT, 'training', 'datasets', 'xgboost')
    train_data_path = os.path.join(data_dir, 'train_data.npz')
    test_data_path = os.path.join(data_dir, 'test_data.npz')

    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"{train_data_path} not found. Run generate_data.py with MODEL_TYPE='xgboost' first.")

    if verbose:
        print("Loading data...")
    train_data = np.load(train_data_path)
    X_train, y_train = train_data['X'], train_data['y']

    test_data = np.load(test_data_path, allow_pickle=True)
    X_test, y_test = test_data['X'], test_data['y']

    if verbose:
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")

    # Copy the params and pop the custom parameter from the dict to avoid passing it to XGBoost constructor
    # Using copy() is essential for rolling windows so we don't permanently alter the imported config!
    xgb_params = XGBOOST_PARAMS.copy()
    use_weights = xgb_params.pop('use_sample_weights', False)
    sample_weights = None

    if use_weights:
        # Calculate sample weights to handle class imbalance
        if verbose:
            print("\nCalculating sample weights for class imbalance...")
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

        if verbose:
            # Print weight distribution for verification
            print("Weight distribution per class:")
            unique_classes = np.unique(y_train)
            for cls in unique_classes:
                # Find the weight for the current class (all samples of a class have the same weight)
                weight_for_class = sample_weights[y_train == cls][0]
                count = (y_train == cls).sum()
                print(f"  Class {int(cls)} (count: {count}): weight = {weight_for_class:.4f}")
    else:
        if verbose:
            print("\nSample weighting is disabled by config.")

    # 2. Configure and Train XGBoost Model
    if verbose:
        print("Configuring XGBoost model...")

    # Check for GPU availability for XGBoost
    try:
        import cupy as cp
        # Force a small kernel operation to verify the CUDA environment is fully functional
        _ = cp.array([1]) + 1
        device = 'cuda'
        if verbose:
            print("Found CuPy and working CUDA environment, moving data to GPU...")
        X_train = cp.array(X_train)
        y_train = cp.array(y_train)
        if use_weights:
            sample_weights = cp.array(sample_weights)
        X_test_gpu = cp.array(X_test)
        y_test_gpu = cp.array(y_test)
        eval_set = [(X_test_gpu, y_test_gpu)]
    except Exception as e:
        device = 'cpu'
        if verbose:
            print(f"GPU acceleration unavailable (CuPy/CUDA error: {e}). Falling back to CPU.")
        X_test_gpu = X_test
        eval_set = [(X_test, y_test)]

    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        device=device,
        **xgb_params
    )

    if verbose:
        print(f"Starting training...")
        
    fit_verbose = 100 if verbose else False
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=eval_set,
        verbose=fit_verbose # Print progress every 100 rounds if verbose, else silent
    )

    # 3. Evaluate Model
    if verbose:
        print("\nEvaluating model on test set...")
    
    y_pred = model.predict(X_test_gpu)    
    if verbose:
        print('using sample weights', use_weights)
        print('max_depth', xgb_params['max_depth'])
        print('colsample_bytree', xgb_params['colsample_bytree'])

    # If using GPU, move predictions back to CPU for sklearn metrics
    if device == 'cuda':
        y_pred = cp.asnumpy(y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    if verbose:
        print(f"Test Accuracy: {accuracy:.4f}")

    # Compute and print Training Accuracy
    y_train_pred = model.predict(X_train)
    if device == 'cuda':
        train_accuracy = accuracy_score(cp.asnumpy(y_train), cp.asnumpy(y_train_pred))
    else:
        train_accuracy = accuracy_score(y_train, y_train_pred)
    if verbose:
        print(f"Train Accuracy: {train_accuracy:.4f}")

    if verbose:
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Down', 'Neutral', 'Up']))

    # 4. Save Model
    model_save_dir = os.path.join(PROJECT_ROOT, 'training', 'models', 'xgboost_vatc')
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, 'model.json')
    model.save_model(model_save_path)
    if verbose:
        print(f"Model saved to {model_save_path}")

    return model

if __name__ == '__main__':
    train_model(verbose=True)