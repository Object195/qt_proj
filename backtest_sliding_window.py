import os
import re
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import plotly.graph_objects as go
import plotly.colors as pcolors
import pickle
import importlib.util
from backtest_visualizer import BacktestVisualizer

def run_sliding_window_backtest(processor_file='feature_processor.pkl'):
    # --- Configuration ---
    plot_shap = False
    plot_truth = True
    truth_plot_mode = 'snr' # 'box', 'point', or 'snr'
    CONFIDENCE_THRESHOLD = 0.5  # Probability required to trigger a Buy/Sell signal
    USE_ADJUSTED_PLOT = False   # Toggle to use adjusted predictions for visualization and equity
    
    # --- Locate Sliding Window Results ---
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sliding_window_results')
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} not found. Run run_sliding_window.py first.")
        return

    iter_dirs = [d for d in os.listdir(base_dir) if d.startswith('iter_')]
    # Sort directories strictly by iteration integer
    iter_dirs.sort(key=lambda x: int(re.search(r'iter_(\d+)', x).group(1)))

    if not iter_dirs:
        print(f"No iteration directories found in {base_dir}.")
        return
        
    print(f"Found {len(iter_dirs)} iterations. Processing...")

    # Determine config and NDAYS from the first iteration
    first_config_path = os.path.join(base_dir, iter_dirs[0], 'model', 'config_copy.py')
    if os.path.exists(first_config_path):
        spec = importlib.util.spec_from_file_location("config_model", first_config_path)
        config_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_model)
        PIPELINE = config_model.PIPELINE
        NDAYS = PIPELINE.get('forecast_horizon', 5)
    else:
        print("Config not found, defaulting NDAYS to 5.")
        NDAYS = 5

    try:
        with open(processor_file, 'rb') as f:
            processor = pickle.load(f)
        feature_names = processor.feature_names
    except FileNotFoundError:
        print("feature_processor.pkl not found. Cannot calculate feature importance.")
        processor = None
        feature_names = []

    all_dates = []
    all_probs = []
    all_y_true = []
    all_X = []
    all_shap_0 = []
    all_shap_1 = []
    all_shap_2 = []

    # --- Iterate through Models and Accumulate Test Predictions/SHAP ---
    for iter_dir in iter_dirs:
        model_path = os.path.join(base_dir, iter_dir, 'model', 'model.json')
        data_path = os.path.join(base_dir, iter_dir, 'data', 'test_data.npz')
        
        if not os.path.exists(model_path) or not os.path.exists(data_path):
            print(f"  Missing files in {iter_dir}, skipping.")
            continue

        # Load Data
        with np.load(data_path, allow_pickle=True) as test_data:
            X_test = test_data['X']
            y_test = test_data['y']
            dates_test = test_data['dates']
        
        if len(X_test) == 0:
            continue

        # Load Model
        model = xgb.XGBClassifier()
        model.load_model(model_path)

        # Predict Probabilities
        probs = model.predict_proba(X_test)
        
        # Compute SHAP Values
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_test)
        
        # Accumulate metrics
        all_dates.append(dates_test)
        all_probs.append(probs)
        all_y_true.append(y_test)
        all_X.append(X_test)

        # Accumulate SHAP arrays depending on library version shape
        if isinstance(shap_vals, list):
            all_shap_0.append(shap_vals[0])
            all_shap_1.append(shap_vals[1])
            all_shap_2.append(shap_vals[2])
        elif len(shap_vals.shape) == 3:
            all_shap_0.append(shap_vals[:, :, 0])
            all_shap_1.append(shap_vals[:, :, 1])
            all_shap_2.append(shap_vals[:, :, 2])

    # --- Combine and Deduplicate ---
    print("\nCombining and deduplicating continuous dataset...")
    dates_comb = np.concatenate(all_dates)
    probs_comb = np.concatenate(all_probs, axis=0)
    y_comb = np.concatenate(all_y_true)
    X_comb = np.concatenate(all_X, axis=0)
    
    shap_comb = [
        np.concatenate(all_shap_0, axis=0),
        np.concatenate(all_shap_1, axis=0),
        np.concatenate(all_shap_2, axis=0)
    ]

    # Deduplicate based on dates to avoid overlap bounds
    df_meta = pd.DataFrame({'ds': pd.to_datetime(dates_comb)})
    valid_indices = df_meta.drop_duplicates(subset=['ds'], keep='first').index.values
    
    dates_comb = dates_comb[valid_indices]
    probs_comb = probs_comb[valid_indices]
    y_comb = y_comb[valid_indices]
    X_comb = X_comb[valid_indices]
    shap_comb = [s[valid_indices] for s in shap_comb]
    
    predicted_labels = np.argmax(probs_comb, axis=1)
    adjusted_predicted_labels = np.ones_like(predicted_labels)
    adjusted_predicted_labels[(predicted_labels == 0) & (probs_comb[:, 0] > CONFIDENCE_THRESHOLD)] = 0
    adjusted_predicted_labels[(predicted_labels == 2) & (probs_comb[:, 2] > CONFIDENCE_THRESHOLD)] = 2

    df_combined = pd.DataFrame({
        'ds': dates_comb,
        'Prob_Down': probs_comb[:, 0],
        'Prob_Neutral': probs_comb[:, 1],
        'Prob_Up': probs_comb[:, 2],
        'True_Label': y_comb,
        'Predicted_Label': predicted_labels,
        'Adjusted_Predicted_Label': adjusted_predicted_labels
    }).sort_values('ds')

    # --- Evaluate Visuals and Equidity ---
    print("\nEvaluating combined performance...")
    df_prices = pd.read_csv('processed_data.csv')
    visualizer = BacktestVisualizer(df_combined, df_prices)
    
    visualizer.plot(title_suffix="Sliding Window Combined Backtest", use_adjusted=USE_ADJUSTED_PLOT, ndays=NDAYS)

    # --- SHAP and Truth Value Analysis ---
    if processor is not None and feature_names:
        if USE_ADJUSTED_PLOT:
            non_neutral_mask = adjusted_predicted_labels != 1
            decision_type = "Adjusted Non-Neutral"
        else:
            non_neutral_mask = predicted_labels != 1
            decision_type = "Non-Neutral"
        
        if np.sum(non_neutral_mask) > 0:
            shap_nn = [s[non_neutral_mask] for s in shap_comb]
            y_nn = y_comb[non_neutral_mask]
            
            mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_nn], axis=0)
            
            shap_importance_dict = {}
            for i, shap_val in enumerate(mean_abs_shap):
                if i < len(feature_names):
                    shap_importance_dict[feature_names[i]] = float(shap_val)
            
            shap_group_series = processor.calculate_grouped_importance(shap_importance_dict)
            sorted_shap_groups = shap_group_series.index.tolist()
            
            if plot_shap:   
                fig_shap = go.Figure()
                for group in sorted_shap_groups:
                    group_feats = processor.feature_groups[group]
                    importance_data = {feat: shap_importance_dict.get(feat, 0.0) for feat in group_feats}
                    group_series = pd.Series(importance_data).sort_values(ascending=False)
                    group_series = group_series[group_series > 0]
                    
                    if not group_series.empty:
                        fig_shap.add_trace(go.Bar(
                            x=[[group] * len(group_series), group_series.index],
                            y=group_series.values,
                            name=group
                        ))
                        
                fig_shap.update_layout(
                    title=f'Combined Feature Importance (Mean |SHAP| for {decision_type} Decisions)',
                    yaxis_title='Mean |SHAP| Value',
                    template='plotly_dark',
                    height=700,
                    xaxis_tickangle=-90,
                    showlegend=True,
                    legend_title_text="Feature Groups"
                )
                fig_shap.show()

            if plot_truth:
                raw_shap_directional = shap_nn[2] - shap_nn[0]
                y_dir = y_nn - 1 
                truth_values = raw_shap_directional * y_dir[:, np.newaxis]
                
                fig_truth = go.Figure()
                max_shap = max(shap_importance_dict.values()) if shap_importance_dict else 1.0
                if max_shap == 0: max_shap = 1.0
                
                for group in sorted_shap_groups:
                    group_feats = processor.feature_groups[group]
                    importance_data = {feat: shap_importance_dict.get(feat, 0.0) for feat in group_feats}
                    group_series = pd.Series(importance_data).sort_values(ascending=False)
                    group_series = group_series[group_series > 0]
                    
                    for feat in group_series.index:
                        if feat in feature_names:
                            feat_idx = feature_names.index(feat)
                            feat_truth = truth_values[:, feat_idx]
                            
                            intensity = shap_importance_dict[feat] / max_shap
                            color = pcolors.sample_colorscale('YlOrRd', [intensity])[0]
                            
                            mean_truth = np.mean(feat_truth)
                            std_truth = np.std(feat_truth)
                            snr = mean_truth / std_truth if std_truth > 1e-9 else 0.0
                            fig_truth.add_trace(go.Bar(y=[snr], x=[[group], [feat]], name=feat, marker_color=color, showlegend=False))
                
                fig_truth.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(colorscale='YlOrRd', cmin=0, cmax=max_shap, colorbar=dict(title="Mean |SHAP|"), showscale=True), showlegend=False, hoverinfo='none'))
                fig_truth.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                fig_truth.update_layout(title=f'Combined Directional SHAP Truth Value SNR', yaxis_title='SNR (Mean / Std Dev)', template='plotly_dark', height=700, xaxis_tickangle=-90)
                fig_truth.show()

if __name__ == '__main__':
    run_sliding_window_backtest("feature_processor_shap.pkl")
   #run_sliding_window_backtest("feature_processor.pkl")