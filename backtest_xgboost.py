#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
import importlib.util
import xgboost as xgb
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pcolors
import pickle
import shap
from features.feature_processor import FeatureProcessor
# from config import PIPELINE, FEATURES # This will be replaced by dynamic import
from backtest_visualizer import BacktestVisualizer

def run(train_start, train_end, test_start, test_end):
    # --- Dynamic Config Loading ---
    # 1. Define model paths and load the associated config
    plot_signal = True
    plot_gain_group = False
    plot_gain_individual = False
    plot_shap = False
    plot_truth = True
    truth_plot_mode = 'snr' # 'box', 'point', or 'snr'
    model_dir = 'training/models/xgboost_vatc'
    model_path = os.path.join(model_dir, 'model.json')
    config_path = os.path.join(model_dir, 'config_copy.py')
    processor_file = 'temp_data/feature_processor.pkl'
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config copy not found at {config_path}. "
                                "Please ensure a model has been trained and the config was saved.")

    spec = importlib.util.spec_from_file_location("config_model", config_path)
    config_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_model)
    PIPELINE = config_model.PIPELINE
    FEATURES = config_model.FEATURES
    print(f"Loaded configuration from {config_path}")

    # 2. Load Model and Data
    train_data_path = 'training/datasets/xgboost/train_data.npz'
    test_data_path = 'training/datasets/xgboost/test_data.npz'

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found. Run training/train_xgboost.py first.")

    print("Loading XGBoost model...")
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    print("Loading datasets...")
    train_data = np.load(train_data_path, allow_pickle=True)
    test_data = np.load(test_data_path, allow_pickle=True)

    X_all = np.concatenate([train_data['X'], test_data['X']], axis=0)
    y_all = np.concatenate([train_data['y'], test_data['y']], axis=0)
    dates_all = np.concatenate([train_data['dates'], test_data['dates']])

    # Check for GPU availability
    try:
        import cupy as cp
        _ = cp.array([1]) + 1
        device = 'cuda'
        print("Found CuPy and working CUDA environment. Will use GPU for inference.")
    except Exception:
        device = 'cpu'
        print("GPU acceleration unavailable. Using CPU for inference.")

    # 3. Selection: Choose 'full', 'train', 'test', or 'custom'
    VIEW_MODE = 'test' 
    #VIEW_MODE = 'custom'
    CUSTOM_START = '2023-06-01'
    CUSTOM_END = '2024-06-01'

    CONFIDENCE_THRESHOLD = 0.5  # Probability required to trigger a Buy/Sell signal
    USE_ADJUSTED_PLOT = False     # Toggle to use adjusted predictions for visualization and equity
    NDAYS = PIPELINE.get('forecast_horizon', 5)
    
    train_start = pd.to_datetime(train_start)
    train_end = pd.to_datetime(train_end)
    test_start = pd.to_datetime(test_start)
    test_end = pd.to_datetime(test_end)

    dates_pd = pd.to_datetime(dates_all)
    if VIEW_MODE == 'train':
        mask = (dates_pd >= train_start) & (dates_pd < train_end)
        title_suffix = f"XGBoost - Training Set ({train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')})"
    elif VIEW_MODE == 'test':
        mask = (dates_pd >= test_start) & (dates_pd < test_end)
        title_suffix = f"XGBoost - Test Set ({test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')})"
    elif VIEW_MODE == 'custom':
        mask = (dates_pd >= pd.to_datetime(CUSTOM_START)) & (dates_pd <= pd.to_datetime(CUSTOM_END))
        title_suffix = f"XGBoost - Custom Range ({CUSTOM_START} to {CUSTOM_END})"
    else:
        mask = np.ones(len(dates_all), dtype=bool)
        title_suffix = "XGBoost - Full Dataset"

    X_eval = X_all[mask]
    y_eval = y_all[mask]
    dates_eval = dates_all[mask]

    print(f"Running inference on {len(X_eval)} samples ({VIEW_MODE})...")

    # Move data to GPU if available
    if device == 'cuda':
        X_eval = cp.array(X_eval)

    # Get probabilities and predicted labels
    all_probs = model.predict_proba(X_eval)
    predicted_labels = np.argmax(all_probs, axis=1)

    # If using GPU, move results back to CPU
    if device == 'cuda':
        all_probs = cp.asnumpy(all_probs)
        predicted_labels = cp.asnumpy(predicted_labels)

    # Calculate adjusted predictions mapping low-confidence guesses to Neutral (1)
    adjusted_predicted_labels = np.ones_like(predicted_labels)
    adjusted_predicted_labels[(predicted_labels == 0) & (all_probs[:, 0] > CONFIDENCE_THRESHOLD)] = 0
    adjusted_predicted_labels[(predicted_labels == 2) & (all_probs[:, 2] > CONFIDENCE_THRESHOLD)] = 2

    # 3. Prepare DataFrame for Visualization
    df_prices = pd.read_csv('temp_data/processed_data.csv')

    results_df = pd.DataFrame({
        'ds': dates_eval,
        'Prob_Down': all_probs[:, 0],
        'Prob_Neutral': all_probs[:, 1],
        'Prob_Up': all_probs[:, 2],
        'True_Label': y_eval,
        'Predicted_Label': predicted_labels,
        'Adjusted_Predicted_Label': adjusted_predicted_labels
    })

    # 4. Candlestick Visualization
    visualizer = BacktestVisualizer(results_df, df_prices)
    stats_text = visualizer._calculate_stats(use_adjusted=USE_ADJUSTED_PLOT, ndays=NDAYS)
    if plot_signal:
        
        visualizer.plot(title_suffix=title_suffix, use_adjusted=USE_ADJUSTED_PLOT, ndays=NDAYS)

    # 5. Feature Importance Analysis
    print("\nCalculating and plotting feature importance...")

    try:
        with open(processor_file, 'rb') as f:
            processor = pickle.load(f)
        feature_names = processor.feature_names
    except FileNotFoundError:
        print("feature_processor.pkl not found. Cannot calculate feature importance.")
        processor = None
        feature_names = []

    if processor is not None and feature_names:
        booster = model.get_booster()
        importance = booster.get_score(importance_type='gain')
        
        feature_importance_dict = {}
        for f_idx_str, gain in importance.items():
            f_idx = int(f_idx_str.lstrip('f'))
            if f_idx < len(feature_names):
                feature_importance_dict[feature_names[f_idx]] = gain
                
        importance_series = processor.calculate_grouped_importance(feature_importance_dict)

       
        if plot_gain_group:
            # Create Plotly figure for grouped importance
            fig_grouped = go.Figure()
            fig_grouped.add_trace(go.Bar(
                x=importance_series.index,
                y=importance_series.values,
                marker_color='skyblue',
                name='Group Importance'
            ))
            fig_grouped.add_annotation(
                x=0.98, y=0.98,
                xref="paper", yref="paper",
                text=stats_text.replace('\n', '<br>'), # Use <br> for newlines in plotly
                showarrow=False,
                align='right',
                bordercolor="black",
                borderwidth=1,
                bgcolor="rgba(255, 255, 255, 0.8)"
            )
            fig_grouped.update_layout(
                title='Grouped Feature Importance (Normalized Total Gain)',
                xaxis_title='Feature Group',
                yaxis_title='Normalized Total Gain',
                template='plotly_dark',
                xaxis_tickangle=-45,
                height=600
            )
            fig_grouped.show()
        if plot_gain_individual:
            # 6. Detailed Feature Importance for all groups, grouped by type
            print("\nPlotting detailed feature importance for all features, grouped by type...")
            fig_detailed = go.Figure()

            # Order groups by their total importance (descending), matching the first plot
            sorted_groups = importance_series.index.tolist()

            for group in sorted_groups:
                group_feats = processor.feature_groups[group]
                importance_data = {feat: feature_importance_dict.get(feat, 0.0) for feat in group_feats}
                group_series = pd.Series(importance_data).sort_values(ascending=False)
                group_series = group_series[group_series > 0]
                
                if not group_series.empty:
                    fig_detailed.add_trace(go.Bar(
                        x=[[group] * len(group_series), group_series.index], # Multi-category x-axis
                        y=group_series.values,
                        name=group
                    ))

            fig_detailed.update_layout(
                title='Individual Feature Importance (Gain)',
                yaxis_title='Normalized Total Gain',
                template='plotly_dark',
                height=700,
                xaxis_tickangle=-90,
                showlegend=True,
                legend_title_text="Feature Groups"
            )
            fig_detailed.show()

        # 7. SHAP Value Analysis
        if plot_shap or plot_truth:
            print("\nCalculating SHAP values for non-neutral predictions...")
                
            # Ensure X is on CPU for SHAP explainer
            if device == 'cuda':
                X_eval_cpu = cp.asnumpy(X_eval)
            else:
                X_eval_cpu = X_eval
                
            # Isolate the data where the model made an active Buy/Sell decision
            if USE_ADJUSTED_PLOT:
                non_neutral_mask = adjusted_predicted_labels != 1
                decision_type = "Adjusted Non-Neutral"
            else:
                non_neutral_mask = predicted_labels != 1
                decision_type = "Non-Neutral"
                
            X_eval_nn = X_eval_cpu[non_neutral_mask]
            
            if len(X_eval_nn) > 0:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_eval_nn)
                
                # Handle multiclass output from SHAP (can be a list of arrays or a 3D array depending on the version)
                if isinstance(shap_values, list):
                    # Average across classes and samples
                    mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
                elif len(shap_values.shape) == 3:
                    # (samples, features, classes)
                    mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
                else:
                    mean_abs_shap = np.abs(shap_values).mean(axis=0)
                    
                shap_importance_dict = {}
                for i, shap_val in enumerate(mean_abs_shap):
                    if i < len(feature_names):
                        shap_importance_dict[feature_names[i]] = float(shap_val)
                
                # Calculate group importance based on SHAP to sort the groups natively
                shap_group_series = processor.calculate_grouped_importance(shap_importance_dict)
                sorted_shap_groups = shap_group_series.index.tolist()
                
                # Plot 1: Mean |SHAP| Bar Plot
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
                        title=f'Individual Feature Importance (Mean |SHAP| for {decision_type} Decisions)',
                        yaxis_title='Mean |SHAP| Value',
                        template='plotly_dark',
                        height=700,
                        xaxis_tickangle=-90,
                        showlegend=True,
                        legend_title_text="Feature Groups"
                    )
                    fig_shap.show()

                # Plot 2: Truth Value Plot
                if plot_truth:
                    print(f"\nCalculating and plotting SHAP directional truth values (mode: {truth_plot_mode})...")
                    
                    # Directional SHAP: Net push towards 'Up' (Class 2) vs 'Down' (Class 0)
                    if isinstance(shap_values, list):
                        raw_shap_directional = shap_values[2] - shap_values[0]
                    elif len(shap_values.shape) == 3:
                        raw_shap_directional = shap_values[:, :, 2] - shap_values[:, :, 0]
                    else:
                        raw_shap_directional = shap_values
                    
                    y_eval_nn = y_eval[non_neutral_mask]
                    y_dir = y_eval_nn - 1 
                    truth_values = raw_shap_directional * y_dir[:, np.newaxis]
                    # Get the TRUE labels for the days where the model made a non-neutral prediction
                    true_labels_for_nn_preds = y_eval[non_neutral_mask]
                    # Convert true labels (0-Down, 1-Neutral, 2-Up) to true direction (-1, 0, 1) for calculation
                    true_direction = true_labels_for_nn_preds - 1
                    # The "truth value" is the directional SHAP value multiplied by the true direction.
                    # A positive result means the feature's contribution (as measured by SHAP) correctly aligned with the actual market outcome.
                    truth_values = raw_shap_directional * true_direction[:, np.newaxis]
                    
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
                                
                                if truth_plot_mode == 'box':
                                    fig_truth.add_trace(go.Box(y=feat_truth, x=[[group] * len(feat_truth), [feat] * len(feat_truth)], name=feat, marker_color=color, showlegend=False, boxpoints=False))
                                elif truth_plot_mode == 'point':
                                    mean_truth = np.mean(feat_truth)
                                    std_truth = np.std(feat_truth)
                                    fig_truth.add_trace(go.Scatter(y=[mean_truth], x=[[group], [feat]], mode='markers', name=feat, marker=dict(color=color, size=8, line=dict(width=1, color='DarkSlateGrey')), error_y=dict(type='data', array=[std_truth], visible=True, thickness=1, color='rgba(200, 200, 200, 0.3)'), showlegend=False))
                                elif truth_plot_mode == 'snr':
                                    mean_truth = np.mean(feat_truth)
                                    std_truth = np.std(feat_truth)
                                    snr = mean_truth / std_truth if std_truth > 1e-9 else 0.0
                                    fig_truth.add_trace(go.Bar(y=[snr], x=[[group], [feat]], name=feat, marker_color=color, showlegend=False))
                    
                    fig_truth.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(colorscale='YlOrRd', cmin=0, cmax=max_shap, colorbar=dict(title="Mean |SHAP|"), showscale=True), showlegend=False, hoverinfo='none'))
                    fig_truth.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
                    
                    if truth_plot_mode == 'box':
                        title_text, yaxis_title = f'Directional SHAP Truth Value Distribution (for {decision_type} Decisions)', 'Truth Value (Direction * Net SHAP)'
                    elif truth_plot_mode == 'point':
                        title_text, yaxis_title = f'Directional SHAP Truth Value Mean & Std Dev (for {decision_type} Decisions)', 'Truth Value (Direction * Net SHAP)'
                    elif truth_plot_mode == 'snr':
                        title_text, yaxis_title = f'Directional SHAP Truth Value SNR (for {decision_type} Decisions)', 'SNR (Mean / Std Dev)'

                    fig_truth.update_layout(title=title_text, yaxis_title=yaxis_title, template='plotly_dark', height=700, xaxis_tickangle=-90)
                    fig_truth.show()
            else:
                print("No non-neutral predictions found to calculate SHAP values.")

if __name__ == '__main__':
    # Import dates dynamically from run_pipeline to ensure consistency
    try:
        import run_pipeline
        run(run_pipeline.TRAIN_START, run_pipeline.TRAIN_END, run_pipeline.TEST_START, run_pipeline.TEST_END)
    except ImportError:
        print("Warning: Could not import run_pipeline. Using default dates.")
        run('2020-01-01', '2023-01-01', '2023-03-01', '2024-03-01')
# %%