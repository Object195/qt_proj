#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
import importlib.util
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
# from config import PIPELINE, FEATURES # This will be replaced by dynamic import
from backtest_visualizer import BacktestVisualizer

# --- Dynamic Config Loading ---
# 1. Define model paths and load the associated config
model_dir = 'training/models/xgboost_vatc'
model_path = os.path.join(model_dir, 'model.json')
config_path = os.path.join(model_dir, 'config_copy.py')

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

#%%
# 3. Selection: Choose 'full', 'train', 'test', or 'custom'
VIEW_MODE = 'test' 
#VIEW_MODE = 'custom'
CUSTOM_START = '2023-06-01'
CUSTOM_END = '2024-06-01'

train_start = pd.to_datetime(PIPELINE['train_start_date'])
train_end = pd.to_datetime(PIPELINE['train_end_date'])
test_end = pd.to_datetime(PIPELINE['fetch_end_date'])

dates_pd = pd.to_datetime(dates_all)
if VIEW_MODE == 'train':
    mask = (dates_pd >= train_start) & (dates_pd < train_end)
    title_suffix = f"XGBoost - Training Set ({PIPELINE['train_start_date']} to {PIPELINE['train_end_date']})"
elif VIEW_MODE == 'test':
    mask = (dates_pd >= train_end) & (dates_pd < test_end)
    title_suffix = f"XGBoost - Test Set ({PIPELINE['train_end_date']} to {PIPELINE['fetch_end_date']})"
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

# 3. Prepare DataFrame for Visualization
df_prices = pd.read_csv('processed_data.csv')

results_df = pd.DataFrame({
    'ds': dates_eval,
    'Prob_Down': all_probs[:, 0],
    'Prob_Neutral': all_probs[:, 1],
    'Prob_Up': all_probs[:, 2],
    'True_Label': y_eval,
    'Predicted_Label': predicted_labels
})

# 4. Candlestick Visualization
visualizer = BacktestVisualizer(results_df, df_prices)
visualizer.plot(title_suffix=title_suffix)
#%%
# 5. Feature Importance Analysis
print("\nCalculating and plotting feature importance...")

# Get original feature names and model parameters from config
feature_names = [f['name'] for f in FEATURES if f['name'] != 'Target_VATC']
window_size = PIPELINE['input_window']
num_original_features = len(feature_names)

# XGBoost provides importance as a dictionary {'f0': gain, 'f1': gain, ...}
# We need to map these flattened feature indices back to our original features.
booster = model.get_booster()
importance = booster.get_score(importance_type='gain')

grouped_importance = {name: 0.0 for name in feature_names}


for f_idx_str, gain in importance.items():
    # The feature index 'f_idx_str' is like 'f123'
    f_idx = int(f_idx_str.lstrip('f'))
    
    # The data is flattened from a (window_size, num_features) matrix.
    # Numpy's flatten is row-major, so the pattern is [feat1_day1, feat2_day1, ..., feat1_day2, ...].
    # The modulo operator correctly maps the flattened index back to the original feature column index.
    original_feature_index = f_idx % num_original_features
    original_feature_name = feature_names[original_feature_index]
    
    grouped_importance[original_feature_name] += gain

# Create a pandas Series for easy sorting and plotting
importance_series = pd.Series(grouped_importance).sort_values(ascending=False)
#%%
# Normalize by sum of all gains
total_gain_sum = importance_series.sum()
if total_gain_sum > 0:
    importance_series = importance_series / total_gain_sum

importance_series = importance_series[importance_series > 0]
delta = np.abs(predicted_labels - y_eval)
total = len(delta)
stats_text = f"Δ=0:{(delta==0).sum()/total:.1%}, Δ=1:{(delta==1).sum()/total:.1%}, Δ=2:{(delta==2).sum()/total:.1%}"
plt.figure()
plt.bar(importance_series.index, importance_series.values)
plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5))
plt.grid()
plt.xticks(rotation=90)
plt.show()
#plt.tight_layout()
'''
# Plotting with Plotly
fig = go.Figure([go.Bar(x=importance_series.index, y=importance_series.values, marker_color='#636EFA')])
fig.update_layout(
    title='Grouped Feature Importance (Normalized Total Gain)',
    xaxis_title='Feature',
    yaxis_title='Normalized Total Gain',
    #yaxis=dict(type='log'),
    template='plotly_dark'
)
fig.show()
'''
# %%