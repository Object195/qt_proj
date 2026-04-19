import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from backtest_metrics import BacktestMetrics
from config import PIPELINE

def analyze_features(start_date=None, end_date=None, method='spearman', clip_percentile=0.95, top_n=15):
    data_file = 'temp_data/feature_set.csv'
    processor_file = 'temp_data/feature_processor.pkl'
    target_col = 'log_return_target'

    if not os.path.exists(data_file) or not os.path.exists(processor_file):
        raise FileNotFoundError(f"Missing {data_file} or {processor_file}. Please run dataset generation first.")

    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    
    if start_date is None:
        start_date = PIPELINE.get('train_start_date')
    if end_date is None:
        end_date = PIPELINE.get('train_end_date')
        
    if 'ds' in df.columns:
        df['ds'] = pd.to_datetime(df['ds'])
        if start_date:
            df = df[df['ds'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['ds'] <= pd.to_datetime(end_date)]
            
    print(f"Analyzing data from {start_date} to {end_date} using {method.capitalize()} correlation...")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the dataset.")
        
    print(f"Loading FeatureProcessor from {processor_file}...")
    with open(processor_file, 'rb') as f:
        processor = pickle.load(f)
        
    feature_names = processor.feature_names
    feature_groups = processor.feature_groups
    
    print("Calculating IC and IR for features against the target...")
    
    # Dictionary to store structured results for sorting
    group_results = {}
    scatter_data = []
    
    # List of specific features to exclude from the IC test (e.g., long-memory indicators)
    EXCLUDE_FEATURES = ['ema_bias_200', 'ema_bias_100', 'ema_bias_50', 'macd_line']

    # 1. Collect all valid features to be analyzed, respecting the exclusion list
    features_to_analyze = [
        f for f in feature_names
        if f in df.columns and f not in EXCLUDE_FEATURES
    ]
    for f in feature_names:
        if f in EXCLUDE_FEATURES:
            print(f"  - Skipping feature '{f}' (explicitly excluded).")

    # 2. Perform one vectorized calculation for all features
    print(f"Calculating IC/IR for {len(features_to_analyze)} features...")
    ic_df, ir_df = BacktestMetrics.calculate_rolling_ic_ir(
        df, features_to_analyze, target_col,
        ic_window=64, ir_window=128,
        method=method, clip_percentile=clip_percentile, stride=PIPELINE.get('forecast_horizon')
    )

    # 3. Process the results from the returned DataFrames
    for group, feats in feature_groups.items():
        # Consider only features that were actually processed
        valid_feats_in_group = [f for f in feats if f in features_to_analyze]
        if not valid_feats_in_group:
            continue
            
        group_results[group] = []
        for f in valid_feats_in_group:
            ic = ic_df[f]
            ir = ir_df[f]
            
            # Extract clean numpy arrays without NaNs for visualization
            clean_ic = ic.dropna().values
            clean_ir = ir.dropna().values
            
            if len(clean_ic) > 0:
                group_results[group].append({
                    'feature': f,
                    'median_abs_ic': np.median(np.abs(clean_ic)),
                    'ic': clean_ic,
                    'ir': clean_ir
                })
                scatter_data.append({
                    'feature': f,
                    'group': group,
                    'mean_ic': np.mean(clean_ic),
                    'mean_ir': np.mean(clean_ir) if len(clean_ir) > 0 else np.nan
                })
                
    print(f"\n--- Top {top_n} Features by Absolute Mean IC ---")
    top_features = sorted(scatter_data, key=lambda x: abs(x['mean_ic']), reverse=True)[:top_n]
    for i, item in enumerate(top_features, 1):
        print(f"{i:2d}. {item['feature']:<30} | Mean IC: {item['mean_ic']:>7.4f} | Mean IR: {item['mean_ir']:>7.4f}")
    print("-" * 65 + "\n")
            
    # Create Plotly Subplots
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Information Coefficient (IC)", "Information Ratio (IR)")
    )
    
    print("Generating Box Plots...")
    for group, results in group_results.items():
        # Sort features inside each group descending by their median |IC|
        results.sort(key=lambda x: x['median_abs_ic'], reverse=True)
        
        for res in results:
            feat_name = res['feature']
            
            # FIX 1: Use a multi-category x-axis [group, feature]. 
            # This gives each feature its own dedicated slot, preventing Plotly 
            # from reserving empty space for all other traces at the same tick.
            trace_kwargs = dict(
                x=[[group] * len(res['ic']), [feat_name] * len(res['ic'])], 
                name=feat_name, 
                boxpoints=False, 
                legendgroup=group # Groups the legend items cleanly
            )
            
            fig.add_trace(go.Box(y=res['ic'], showlegend=True, **trace_kwargs), row=1, col=1)
            fig.add_trace(go.Box(y=res['ir'], showlegend=False, **trace_kwargs), row=2, col=1)
            
    fig.update_layout(
        title=f"Feature Importance Analysis: Rolling {method.capitalize()} IC & IR vs Target ({start_date} to {end_date}, Clipped {clip_percentile*100:g}%)",
        # FIX 2: Remove boxmode='group', boxgap, and boxgroupgap. 
        # The multi-category axis handles the grouping and spacing perfectly on its own.
        template='plotly_dark',
        height=900,
        xaxis2_title="Feature Groups & Individual Indicators (Sorted descending by |IC| median)",
        legend_title_text="Feature Groups"
    )
    
    # FIX 3: Angle the ticks to comfortably fit the new sub-labels
    fig.update_xaxes(tickangle=45) 
    fig.show()

    print("Generating Scatter Plot...")
    scatter_fig = go.Figure()
    
    # Group by 'group' to have legend colors per group matching the box plots
    groups_in_scatter = sorted(list(set([d['group'] for d in scatter_data])))
    for g in groups_in_scatter:
        g_data = [d for d in scatter_data if d['group'] == g]
        scatter_fig.add_trace(go.Scatter(
            x=[d['mean_ic'] for d in g_data],
            y=[d['mean_ir'] for d in g_data],
            mode='markers',
            text=[d['feature'] for d in g_data],
            name=g,
            hoverinfo='text+x+y',
            marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey'))
        ))

    scatter_fig.update_layout(
        title=f"Mean IC vs Mean IR Scatter ({start_date} to {end_date}, Clipped {clip_percentile*100:g}%)",
        xaxis_title="Mean Information Coefficient (IC)",
        yaxis_title="Mean Information Ratio (IR)",
        template='plotly_dark',
        height=800,
        hovermode='closest'
    )
    
    scatter_fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    scatter_fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    scatter_fig.show()

if __name__ == "__main__":
    analyze_features(method='spearman', clip_percentile=0.95, top_n=15)