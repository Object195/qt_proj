import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import os
from backtest_metrics import BacktestMetrics
from config import PIPELINE

# --- FILTER PARAMETERS ---
FILTER_PARAMS_1 = {
    'start_date': '2016-03-01',
    'end_date': '2026-03-01',
    'length': '1024D',           # Rolling window length (e.g., 1 year)
    'gap_interval': '128D',      # Step forward interval (e.g., 90 days)
    'method': 'spearman',
    'clip_percentile': 0.95,
    'data_file': 'feature_set.csv',
    'processor_file': 'feature_processor.pkl',
    'output_processor_file': 'feature_processor_temp.pkl',
    'stride': 1,
    'ic_lim': 0.015,            # Threshold for absolute mean IC
    'ir_lim': 0.25,             # Threshold for absolute mean IR
    'pass_rate_lim': 0.25,      # Minimum proportion of iterations where thresholds must be met
    'mode':  'individual', #'group', #,       # 'individual' or 'group'
    'target_col': 'log_return_target_1',
    'exclude_features': [], #['ema_bias_200', 'ema_bias_100', 'ema_bias_50', 'macd_line'] # Keep long-memory indicators safe
    'generate_plot' : False
}
FILTER_PARAMS_2 = {
    'start_date': '2017-03-01',
    'end_date': '2026-03-01',
    'length': '1024D',           # Rolling window length (e.g., 1 year)
    'gap_interval': '128D',      # Step forward interval (e.g., 90 days)
    'method': 'spearman',
    'clip_percentile': 0.95,
    'data_file': 'feature_set.csv',
    'processor_file': 'feature_processor_temp.pkl',
    'output_processor_file': 'feature_processor.pkl',
    'stride': PIPELINE.get('forecast_horizon'),
    'ic_lim': 0.025,            # Threshold for absolute mean IC
    'ir_lim': 0.25,             # Threshold for absolute mean IR
    'pass_rate_lim': 0.25,      # Minimum proportion of iterations where thresholds must be met
    'mode':  'individual', #'group', #,       # 'individual' or 'group'
    'target_col': 'log_return_target',
    'exclude_features': [], #['ema_bias_200', 'ema_bias_100', 'ema_bias_50', 'macd_line'] # Keep long-memory indicators safe
    'generate_plot' : True
}
def run_feature_filter(params):
    print("--- Starting Rolling Feature Filter ---")
    print(f"Mode: {params['mode'].upper()} | IC Limit: {params['ic_lim']} | IR Limit: {params['ir_lim']}")
    print(f"Window: {params['length']} | Step: {params['gap_interval']}")
    
    if not os.path.exists(params['data_file']) or not os.path.exists(params['processor_file']):
        raise FileNotFoundError("Data file or Processor file missing.")

    # 1. Load Data and Processor
    print(f"\nLoading data from {params['data_file']}...")
    df = pd.read_csv(params['data_file'])
    df['ds'] = pd.to_datetime(df['ds'])
    
    print(f"Loading FeatureProcessor from {params['processor_file']}...")
    with open(params['processor_file'], 'rb') as f:
        processor = pickle.load(f)
        
    feature_names = processor.feature_names
    feature_groups = processor.feature_groups
    
    # Map each feature to its group for later plotting and group-mode logic
    group_map = {}
    for group, feats in feature_groups.items():
        for f in feats:
            group_map[f] = group
            
    features_to_analyze = [
        f for f in feature_names 
        if f in df.columns and f not in params['exclude_features']
    ]
    
    # 2. Setup Iteration Variables
    current_start = pd.to_datetime(params['start_date'])
    end_date = pd.to_datetime(params['end_date'])
    length_td = pd.Timedelta(params['length'])
    gap_td = pd.Timedelta(params['gap_interval'])
    
    def parse_interval(interval_str):
        if isinstance(interval_str, str) and interval_str.upper().endswith('M'):
            return pd.DateOffset(months=int(interval_str[:-1]))
        return pd.Timedelta(interval_str)
        
    length_td = parse_interval(params['length'])
    gap_td = parse_interval(params['gap_interval'])
    
    # Store historical metric means for each feature to calculate global avg & std later
    feature_metrics = {f: {'ic_means': [], 'ir_means': []} for f in features_to_analyze}
    
    feature_success_counts = {f: 0 for f in features_to_analyze}
    iteration_count = 0
    
    # 3. Rolling Window Evaluation
    print("\n--- Beginning Rolling Evaluation ---")
    while current_start + length_td <= end_date:
        iteration_count += 1
        current_end = current_start + length_td
        print(f"Iteration {iteration_count}: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")
        
        window_df = df[(df['ds'] >= current_start) & (df['ds'] <= current_end)].copy()
        
        if window_df.empty:
            print("  -> Empty window, skipping.")
            current_start += gap_td
            continue
            
        # Calculate IC/IR for this window
        ic_df, ir_df = BacktestMetrics.calculate_rolling_ic_ir(
            window_df, features_to_analyze, params['target_col'],
            ic_window=64, ir_window=128,
            method=params['method'], clip_percentile=params['clip_percentile'], 
            stride=params['stride']
        )
        
        window_ic_means = {}
        window_ir_means = {}
        
        for f in features_to_analyze:
            clean_ic = ic_df[f].dropna().values
            clean_ir = ir_df[f].dropna().values
            
            m_ic = np.mean(clean_ic) if len(clean_ic) > 0 else 0.0
            m_ir = np.mean(clean_ir) if len(clean_ir) > 0 else 0.0
            
            window_ic_means[f] = m_ic
            window_ir_means[f] = m_ir
            
            feature_metrics[f]['ic_means'].append(m_ic)
            feature_metrics[f]['ir_means'].append(m_ir)
            
        # Determine Success in this iteration
        passed_in_window = 0
        
        if params['mode'] == 'individual':
            for f in features_to_analyze:
                if abs(window_ic_means[f]) > params['ic_lim'] and abs(window_ir_means[f]) > params['ir_lim']:
                    feature_success_counts[f] += 1
                    passed_in_window += 1
                    
        elif params['mode'] == 'group':
            for group, feats in feature_groups.items():
                valid_feats = [f for f in feats if f in features_to_analyze]
                if not valid_feats:
                    continue
                
                max_ic = max([abs(window_ic_means[f]) for f in valid_feats])
                max_ir = max([abs(window_ir_means[f]) for f in valid_feats])
                
                if max_ic > params['ic_lim'] and max_ir > params['ir_lim']:
                    for f in valid_feats:
                        feature_success_counts[f] += 1
                        passed_in_window += 1
                        
        print(f"  -> Features passing criteria in this window: {passed_in_window}")
            
        current_start += gap_td

    # 4. Final Processing & Filtering
    if iteration_count == 0:
        print("No iterations completed. Check your start/end dates and window length.")
        return

    print(f"\n--- Iteration Results ({iteration_count} total iterations) ---")
    pass_rate_lim = params.get('pass_rate_lim', 0.5)
    
    final_S = set()
    for f in features_to_analyze:
        rate = feature_success_counts[f] / iteration_count
        if rate < pass_rate_lim:
            final_S.add(f)
            
    print(f"Total features filtered out (pass rate < {pass_rate_lim*100:.1f}%): {len(final_S)}")
    
    if len(final_S) > 0:
        for f in sorted(list(final_S)):
            rate = feature_success_counts[f] / iteration_count
            print(f"  - Removing {f} (Pass Rate: {rate*100:.1f}%)")
            
        # Update processor dictionary and list
        processor.feature_names = [f for f in processor.feature_names if f not in final_S]
        
        for group in processor.feature_groups:
            processor.feature_groups[group] = [f for f in processor.feature_groups[group] if f not in final_S]
            
        # Clean up any empty groups
        processor.feature_groups = {g: fs for g, fs in processor.feature_groups.items() if len(fs) > 0}
        
        # Save updated processor
        print(f"\nSaving filtered processor to {params['output_processor_file']}...")
        with open(params['output_processor_file'], 'wb') as f:
            pickle.dump(processor, f)
    else:
        print("\nNo features were flagged consistently across all iterations. Processor unchanged.")
        # Save it anyway just to have the expected output file
        with open(params['output_processor_file'], 'wb') as f:
            pickle.dump(processor, f)
    if not params['generate_plot']: return
    # 5. Summary Plot (2D Error Bars)
    print("\nGenerating Iteration Summary Plot...")
    scatter_fig = go.Figure()
    
    # Group by original groups for consistent colors
    groups_in_scatter = sorted(list(set([group_map[f] for f in features_to_analyze if f in group_map])))
    
    for g in groups_in_scatter:
        g_feats = [f for f in features_to_analyze if group_map.get(f) == g]
        if not g_feats: continue
        
        avg_ic = [np.mean(feature_metrics[f]['ic_means']) for f in g_feats]
        std_ic = [np.std(feature_metrics[f]['ic_means']) for f in g_feats]
        
        avg_ir = [np.mean(feature_metrics[f]['ir_means']) for f in g_feats]
        std_ir = [np.std(feature_metrics[f]['ir_means']) for f in g_feats]
        
        scatter_fig.add_trace(go.Scatter(
            x=avg_ic,
            y=avg_ir,
            mode='markers',
            text=g_feats,
            name=g,
            error_x=dict(type='data', array=std_ic, visible=True, thickness=1, color='rgba(255,255,255,0.2)'),
            error_y=dict(type='data', array=std_ir, visible=True, thickness=1, color='rgba(255,255,255,0.2)'),
            marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')),
            hoverinfo='text+x+y'
        ))

    scatter_fig.update_layout(
        title=f"Rolling Feature Stability: Avg IC vs Avg IR ({iteration_count} iterations)",
        xaxis_title=f"Average {params['method'].capitalize()} IC (with Std Dev)",
        yaxis_title="Average Information Ratio (IR) (with Std Dev)",
        template='plotly_dark',
        height=900,
        hovermode='closest'
    )
    
    # Add boundary lines
    scatter_fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    scatter_fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)

    scatter_fig.show()
    print("Done.")

def run(start_date, end_date):
    FILTER_PARAMS_1['start_date'] = start_date
    FILTER_PARAMS_1['end_date'] = end_date
    FILTER_PARAMS_2['start_date'] = start_date
    FILTER_PARAMS_2['end_date'] = end_date
    run_feature_filter(FILTER_PARAMS_1)
    run_feature_filter(FILTER_PARAMS_2)

if __name__ == "__main__":
    run('2016-03-01', '2026-03-01')