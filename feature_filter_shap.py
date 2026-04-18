import os
import re
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import pickle

def run_shap_filter(
    #processor_file='feature_processor.pkl', 
    processor_file='feature_processor_shap.pkl',
    output_processor_file='feature_processor_shap.pkl'
):
    print(f"--- Starting SHAP Value Feature Filter ---")
    print(f"Processor In:  {processor_file}")
    print(f"Processor Out: {output_processor_file}")
    print(f"Thresholds:    |SHAP| > Mean |SHAP|, Truth SNR < 0")
    
    # 1. Locate Sliding Window Results
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sliding_window_results')
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found. Run run_sliding_window.py first.")
        return

    iter_dirs = [d for d in os.listdir(base_dir) if d.startswith('iter_')]
    iter_dirs.sort(key=lambda x: int(re.search(r'iter_(\d+)', x).group(1)))

    if not iter_dirs:
        print(f"No iteration directories found in {base_dir}.")
        return

    # 2. Load the specified feature processor
    try:
        with open(processor_file, 'rb') as f:
            processor = pickle.load(f)
        feature_names = processor.feature_names
    except FileNotFoundError:
        print(f"{processor_file} not found. Ensure the file exists.")
        return

    all_dates, all_probs, all_y_true, all_X = [], [], [], []
    all_shap_0, all_shap_1, all_shap_2 = [], [], []

    # 3. Accumulate Data & SHAP Values Iteratively
    print(f"Loading models and extracting SHAP values across {len(iter_dirs)} iterations...")
    for iter_dir in iter_dirs:
        model_path = os.path.join(base_dir, iter_dir, 'model', 'model.json')
        data_path = os.path.join(base_dir, iter_dir, 'data', 'test_data.npz')
        
        if not os.path.exists(model_path) or not os.path.exists(data_path):
            continue

        with np.load(data_path, allow_pickle=True) as test_data:
            X_test, y_test, dates_test = test_data['X'], test_data['y'], test_data['dates']
        
        if len(X_test) == 0: continue

        model = xgb.XGBClassifier()
        model.load_model(model_path)
        probs = model.predict_proba(X_test)
        
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test)
        
        all_dates.append(dates_test)
        all_probs.append(probs)
        all_y_true.append(y_test)
        all_X.append(X_test)

        if isinstance(shap_vals, list):
            all_shap_0.append(shap_vals[0])
            all_shap_1.append(shap_vals[1])
            all_shap_2.append(shap_vals[2])
        elif len(shap_vals.shape) == 3:
            all_shap_0.append(shap_vals[:, :, 0])
            all_shap_1.append(shap_vals[:, :, 1])
            all_shap_2.append(shap_vals[:, :, 2])

    # 4. Deduplicate and Merge
    dates_comb = np.concatenate(all_dates)
    probs_comb = np.concatenate(all_probs, axis=0)
    y_comb = np.concatenate(all_y_true)
    shap_comb = [
        np.concatenate(all_shap_0, axis=0),
        np.concatenate(all_shap_1, axis=0),
        np.concatenate(all_shap_2, axis=0)
    ]

    df_meta = pd.DataFrame({'ds': pd.to_datetime(dates_comb)})
    valid_indices = df_meta.drop_duplicates(subset=['ds'], keep='first').index.values
    
    probs_comb = probs_comb[valid_indices]
    y_comb = y_comb[valid_indices]
    shap_comb = [s[valid_indices] for s in shap_comb]
    predicted_labels = np.argmax(probs_comb, axis=1)

    # 5. Apply Filtering Criteria
    print("\nCalculating metrics and evaluating features...")
    non_neutral_mask = predicted_labels != 1
    if np.sum(non_neutral_mask) == 0:
        print("No non-neutral predictions found to analyze. Exiting.")
        return
        
    shap_nn = [s[non_neutral_mask] for s in shap_comb]
    y_nn = y_comb[non_neutral_mask]
    
    mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_nn], axis=0)
    
    mean_shap_threshold = np.mean(mean_abs_shap)
    print(f"Calculated Mean |SHAP| Threshold: {mean_shap_threshold:.6f}")
    
    raw_shap_directional = shap_nn[2] - shap_nn[0]
    y_dir = y_nn - 1 
    truth_values = raw_shap_directional * y_dir[:, np.newaxis]
    
    features_to_drop = set()
    
    for i, feat in enumerate(feature_names):
        if i >= len(mean_abs_shap): break
        feat_abs_shap = mean_abs_shap[i]
        
        feat_truth = truth_values[:, i]
        mean_truth, std_truth = np.mean(feat_truth), np.std(feat_truth)
        snr = mean_truth / std_truth if std_truth > 1e-9 else 0.0
        
        if feat_abs_shap > mean_shap_threshold and snr < 0:
            print(f"  [-] Dropping {feat} (Mean |SHAP|: {feat_abs_shap:.4f}, SNR: {snr:.4f})")
            features_to_drop.add(feat)

    if not features_to_drop:
        print("\nNo features met the dropping criteria.")
    else:
        print(f"\nFiltered out {len(features_to_drop)} features.")
        processor.feature_names = [f for f in processor.feature_names if f not in features_to_drop]
        for group in processor.feature_groups:
            processor.feature_groups[group] = [f for f in processor.feature_groups[group] if f not in features_to_drop]
        processor.feature_groups = {g: fs for g, fs in processor.feature_groups.items() if len(fs) > 0}

    print(f"Saving new processor object to {output_processor_file}...")
    with open(output_processor_file, 'wb') as f:
        pickle.dump(processor, f)
    print("Done.")

if __name__ == '__main__':
    run_shap_filter()