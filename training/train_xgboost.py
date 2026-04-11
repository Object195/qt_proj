import sys
import os
import shutil
from datetime import datetime
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_sample_weight

# Add the project root to sys.path to allow importing config.py
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from config import XGBOOST_PARAMS, PIPELINE

def run(data_dir=None, model_save_dir=None, verbose=True):
    # 1. Load Data
    data_dir = data_dir if data_dir else os.path.join(PROJECT_ROOT, 'training', 'datasets', 'xgboost')
    train_data_path = os.path.join(data_dir, 'train_data.npz')
    test_data_path = os.path.join(data_dir, 'test_data.npz')

    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"{train_data_path} not found. Run generate_data.py with MODEL_TYPE='xgboost' first.")

    if verbose: print("Loading data...")
    train_data = np.load(train_data_path)
    X_train, y_train = train_data['X'], train_data['y']

    test_data = np.load(test_data_path, allow_pickle=True)
    X_test, y_test = test_data['X'], test_data['y']

    if verbose:
        print(f"Training data shape: {X_train.shape}")
        print(f"Test data shape: {X_test.shape}")

    # Get the custom parameter without modifying the original config dict
    use_weights = XGBOOST_PARAMS.get('use_sample_weights', False)
    xgb_params = {k: v for k, v in XGBOOST_PARAMS.items() if k != 'use_sample_weights'}
    #weight_type = 'balanced'
    weight_type = 'target'
    sample_weights = None
    target_factor = 0.5 # power factor for target weight 
    if use_weights:
        if weight_type == 'balanced':
             sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        else:
            if verbose: print("\nCalculating sample weights based on absolute log_forward_return...")
            dates_train = train_data['dates']
            
            processed_data_path = os.path.join(PROJECT_ROOT, 'processed_data.csv')
            df = pd.read_csv(processed_data_path)
            df['ds'] = pd.to_datetime(df['ds'])
            
            weight_col = 'Target_VATC_raw'
            if weight_col not in df.columns:
                if verbose: print(f"Warning: '{weight_col}' not found in processed_data.csv, trying 'log_return_target' as fallback.")
                weight_col = 'Target_VATC_raw'
                
            df_weights = df.drop_duplicates(subset=['ds']).set_index('ds')[weight_col]
            dates_train_pd = pd.to_datetime(dates_train)
            class_weights = compute_sample_weight(class_weight='balanced', y=y_train)
            # Extract absolute values matching the training dates
            raw_weights = df_weights.loc[dates_train_pd].abs().values
            clip_threshold = np.percentile(raw_weights, 95)
            raw_weights = np.clip(raw_weights, a_min=None, a_max=clip_threshold)
            # Normalize mean to 1
            sample_weights = (raw_weights**target_factor) * class_weights
            sample_weights = sample_weights/ np.mean(sample_weights)
            
            if verbose: print(f"Sample weights calculated. Min: {sample_weights.min():.6e}, Max: {sample_weights.max():.6e}, Sum: {sample_weights.sum():.2f}")
    else:
        if verbose: print("\nSample weighting is disabled by config.")

    # 2. Configure and Train XGBoost Model
    if verbose: print("Configuring XGBoost model...")

    # Check for GPU availability for XGBoost
    try:
        import cupy as cp
        # Force a small kernel operation to verify the CUDA environment is fully functional
        _ = cp.array([1]) + 1
        device = 'cuda'
        if verbose: print("Found CuPy and working CUDA environment, moving data to GPU...")
        X_train = cp.array(X_train)
        y_train = cp.array(y_train)
        if use_weights:
            sample_weights = cp.array(sample_weights)
        X_test_gpu = cp.array(X_test)
        y_test_gpu = cp.array(y_test)
        eval_set = [(X_test_gpu, y_test_gpu)]
    except Exception as e:
        device = 'cpu'
        if verbose: print(f"GPU acceleration unavailable (CuPy/CUDA error: {e}). Falling back to CPU.")
        X_test_gpu = X_test
        eval_set = [(X_test, y_test)]

    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        device=device,
        **xgb_params
    )

    if verbose: print(f"Starting training...")
    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=eval_set,
        verbose=100 if verbose else False
    )

    # 3. Evaluate Model
    if verbose: print("\nEvaluating model on test set...")
    y_pred = model.predict(X_test_gpu)
    if verbose:
        print('using sample weights', use_weights)
        print('max_depth', XGBOOST_PARAMS.get('max_depth'))
        print('colsample_bytree', XGBOOST_PARAMS.get('colsample_bytree'))
    # If using GPU, move predictions back to CPU for sklearn metrics
    if device == 'cuda':
        y_pred = cp.asnumpy(y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    if verbose: print(f"Test Accuracy: {accuracy:.4f}")

    # Compute and print Training Accuracy
    y_train_pred = model.predict(X_train)
    if device == 'cuda':
        train_accuracy = accuracy_score(cp.asnumpy(y_train), cp.asnumpy(y_train_pred))
    else:
        train_accuracy = accuracy_score(y_train, y_train_pred)
    if verbose: print(f"Train Accuracy: {train_accuracy:.4f}")

    if verbose:
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Down', 'Neutral', 'Up']))

    # 4. Save Model
    model_save_dir = model_save_dir if model_save_dir else os.path.join(PROJECT_ROOT, 'training', 'models', 'xgboost_vatc')
    os.makedirs(model_save_dir, exist_ok=True)
    model_save_path = os.path.join(model_save_dir, 'model.json')
    model.save_model(model_save_path)
    if verbose: print(f"Model saved to {model_save_path}")

    # Save a copy of the config file
    if PIPELINE.get('save_config_with_timestamp', False):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_filename = f'config_{timestamp}.py'
    else:
        config_filename = 'config_copy.py'

    config_src = os.path.join(PROJECT_ROOT, 'config.py')
    config_dst = os.path.join(model_save_dir, config_filename)
    shutil.copy2(config_src, config_dst)
    if verbose: print(f"Config saved to {config_dst}")

if __name__ == '__main__':
    run()