#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import plotly.graph_objects as go
import pickle

# Add project root to sys.path to allow importing config
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from features.feature_processor import FeatureProcessor
from config import PIPELINE, FEATURES
from backtest_visualizer import BacktestVisualizer

# --- Model and Data Paths ---
model_dir = os.path.join(PROJECT_ROOT, 'training', 'models', 'xgboost_vatc')
model_path = os.path.join(model_dir, 'model.json')
train_data_path = os.path.join(PROJECT_ROOT, 'training', 'datasets', 'xgboost', 'train_data.npz')
test_data_path = os.path.join(PROJECT_ROOT, 'training', 'datasets', 'xgboost', 'test_data.npz')

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

# --- Date Definitions for Visualization ---
# These dates define the boundaries for 'train' and 'test' view modes.
TRAIN_START_DATE = '2021-02-01'
TRAIN_END_DATE = '2024-02-01'
TEST_START_DATE = '2024-03-01'
TEST_END_DATE = '2025-03-01'

CUSTOM_START = '2023-06-01'
CUSTOM_END = '2024-06-01'

CONFIDENCE_THRESHOLD = 0.5  # Probability required to trigger a Buy/Sell signal
USE_ADJUSTED_PLOT = False     # Toggle to use adjusted predictions for visualization and equity
NDAYS = PIPELINE.get('forecast_horizon', 5)

train_start = pd.to_datetime(TRAIN_START_DATE)
train_end = pd.to_datetime(TRAIN_END_DATE)
test_start = pd.to_datetime(TEST_START_DATE)
test_end = pd.to_datetime(TEST_END_DATE)

dates_pd = pd.to_datetime(dates_all)
if VIEW_MODE == 'train':
    mask = (dates_pd >= train_start) & (dates_pd < train_end) 
    title_suffix = f"XGBoost - Training Set ({TRAIN_START_DATE} to {TRAIN_END_DATE})"
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
df_prices = pd.read_csv('processed_data.csv')

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
visualizer.plot(title_suffix=title_suffix, use_adjusted=USE_ADJUSTED_PLOT, ndays=NDAYS)
#%%
# 5. Feature Importance Analysis
print("\nCalculating and plotting feature importance...")

try:
    with open('feature_processor.pkl', 'rb') as f:
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

    stats_text = visualizer._calculate_stats(use_adjusted=USE_ADJUSTED_PLOT, ndays=NDAYS)

    plt.figure(figsize=(10, 6))
    plt.bar(importance_series.index, importance_series.values, color='skyblue')
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, ha='right', va='top', bbox=dict(facecolor='white', alpha=0.8))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.title('Grouped Feature Importance (Normalized Total Gain)')
    plt.ylabel('Normalized Total Gain')
    plt.tight_layout()
    plt.show()
#%%
    # 6. Detailed Feature Importance for a Specific Group
    selected_group = 'BBP' # Change this to inspect other groups
    print(f"\nPlotting detailed feature importance for selected group: {selected_group}...")
    
    try:
        detailed_series = processor.calculate_individual_importance(feature_importance_dict, selected_group)
        if not detailed_series.empty:
            plt.figure(figsize=(10, 6))
            plt.bar(detailed_series.index, detailed_series.values, color='lightgreen')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45, ha='right')
            plt.title(f'Detailed Feature Importance for Group: {selected_group} (Normalized Total Gain)')
            plt.ylabel('Normalized Total Gain')
            plt.tight_layout()
            plt.show()
        else:
            print(f"No positive gain found for features in group '{selected_group}'.")
    except ValueError as e:
        print(e)
# %%