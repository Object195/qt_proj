#%%
import torch
import torch.nn as nn
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from transformers import PatchTSTConfig, PatchTSTForClassification
import numpy as np
import os
from config import PIPELINE

# 1. Load Model and Data
model_path = 'training/models/patchtst_vatc'
train_data_path = 'training/datasets/train_data.pt'
test_data_path = 'training/datasets/test_data.pt'

if not os.path.exists(model_path):
    raise FileNotFoundError("Model not found. Run training/train_patchtst.py first.")

print("Loading model...")

# 1. Load the configuration (not the weights yet)
config = PatchTSTConfig.from_pretrained(model_path)

# 2. Instantiate a blank model with random weights
model = PatchTSTForClassification(config)

# 3. Define the GlobalAvgPool
class GlobalAvgPool(nn.Module):
    def forward(self, x):
        # Input x shape: (batch_size, num_channels, num_patches, d_model)
        # Output shape: (batch_size, d_model)
        return x.mean(dim=(1, 2))

# 4. Replace the head to perfectly match your trained architecture BEFORE loading weights
model.head = nn.Sequential(
    GlobalAvgPool(),
    nn.Dropout(model.config.dropout),
    nn.Linear(model.config.d_model, model.config.num_classes)
)

# 5. Safely load the trained weights into our correctly-shaped model
weights_path = os.path.join(model_path, "model.safetensors")
if os.path.exists(weights_path):
    from safetensors.torch import load_file
    state_dict = load_file(weights_path)
else:
    weights_path = os.path.join(model_path, "pytorch_model.bin")
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=False)

# We can now use strict=True because the architecture perfectly matches the saved weights!
model.load_state_dict(state_dict, strict=True)
model.eval()

print("Loading datasets...")
train_data = torch.load(train_data_path, weights_only=False)
test_data = torch.load(test_data_path, weights_only=False)

X_all = torch.cat([train_data['X'], test_data['X']], dim=0)
y_all = torch.cat([train_data['y'], test_data['y']], dim=0)
dates_all = np.concatenate([train_data['dates'], test_data['dates']])
#%%
# 2. Selection: Choose 'full', 'train', 'test', or 'custom'
VIEW_MODE = 'test' 
#VIEW_MODE = 'custom'
CUSTOM_START = '2023-06-01'
CUSTOM_END = '2024-06-01'

train_start = pd.to_datetime(PIPELINE['train_start_date'])
train_end = pd.to_datetime(PIPELINE['train_end_date'])
test_end = pd.to_datetime(PIPELINE['fetch_end_date'])

dates_pd = pd.to_datetime(dates_all)
if VIEW_MODE == 'train':
    mask = (dates_pd >= train_start) & (dates_pd <= train_end)
    title_suffix = f"Training Set ({PIPELINE['train_start_date']} to {PIPELINE['train_end_date']})"
elif VIEW_MODE == 'test':
    mask = (dates_pd > train_end) & (dates_pd <= test_end)
    title_suffix = f"Test Set ({PIPELINE['train_end_date']} to {PIPELINE['fetch_end_date']})"
elif VIEW_MODE == 'custom':
    mask = (dates_pd >= pd.to_datetime(CUSTOM_START)) & (dates_pd <= pd.to_datetime(CUSTOM_END))
    title_suffix = f"Custom Range ({CUSTOM_START} to {CUSTOM_END})"
else:
    mask = np.ones(len(dates_all), dtype=bool)
    title_suffix = "Full Dataset"

X_eval = X_all[mask]
y_eval = y_all[mask]
dates_eval = dates_all[mask]

print(f"Running inference on {len(X_eval)} samples ({VIEW_MODE})...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

batch_size = 64
probs_list = []

with torch.no_grad():
    for i in range(0, len(X_eval), batch_size):
        batch_X = X_eval[i:i+batch_size].to(device)
        outputs = model(past_values=batch_X)
        # Apply Softmax to get probabilities
        logits = outputs.prediction_logits
        
        # Squeeze sequence dim if present (fallback for different head configurations)
        if logits.dim() == 3 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
            
        probs = torch.softmax(logits, dim=1)
        probs_list.append(probs.cpu().numpy())

all_probs = np.concatenate(probs_list, axis=0)
predicted_labels = np.argmax(all_probs, axis=1)

# Calculate Statistics
delta = np.abs(predicted_labels - y_eval.numpy())
total_points = len(delta)
perf_0 = (delta == 0).sum() / total_points
perf_1 = (delta == 1).sum() / total_points
perf_2 = (delta == 2).sum() / total_points

stats_str = f"Perfect (Δ=0): {perf_0:.2%}, Not Good (Δ=1): {perf_1:.2%}, Bad (Δ=2): {perf_2:.2%}"
print(f"Performance Stats: {stats_str}")

# 3. Prepare DataFrame for Visualization
# Load original processed data to get actual prices corresponding to the dates
df_prices = pd.read_csv('processed_data.csv')
df_prices['ds'] = pd.to_datetime(df_prices['ds'])

# Create a results dataframe
results_df = pd.DataFrame({
    'ds': dates_eval,
    'Prob_Down': all_probs[:, 0],
    'Prob_Neutral': all_probs[:, 1],
    'Prob_Up': all_probs[:, 2],
    'True_Label': y_eval.numpy(),
    'Predicted_Label': predicted_labels
})

# Merge with price data to get Open/High/Low/Close for the specific dates
plot_df = pd.merge(results_df, df_prices, on='ds', how='inner')
plot_df = plot_df.sort_values('ds')

# 4. Visualization
fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.05,
    row_heights=[0.6, 0.4],
    subplot_titles=("Price & Signals", "Model Probabilities")
)

# Row 1: Candlestick
fig.add_trace(go.Candlestick(
    x=plot_df['ds'],
    open=plot_df['open'], high=plot_df['high'],
    low=plot_df['low'], close=plot_df['close'],
    name="Price"
), row=1, col=1)

# Add Ground Truth Signals (Actual)
gt_buys = plot_df[plot_df['True_Label'] == 2]
gt_sells = plot_df[plot_df['True_Label'] == 0]

fig.add_trace(go.Scatter(
    x=gt_buys['ds'], y=gt_buys['low'] * 0.96,
    mode='markers', name='Actual Buy',
    marker=dict(symbol='triangle-up', size=10, color='#00ff00')
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=gt_sells['ds'], y=gt_sells['high'] * 1.04,
    mode='markers', name='Actual Sell',
    marker=dict(symbol='triangle-down', size=10, color='#ff0000')
), row=1, col=1)

# Add Predicted Signals
pred_buys = plot_df[plot_df['Predicted_Label'] == 2]
pred_sells = plot_df[plot_df['Predicted_Label'] == 0]

fig.add_trace(go.Scatter(
    x=pred_buys['ds'], y=pred_buys['low'] * 0.98,
    mode='markers', name='Predicted Buy',
    marker=dict(symbol='triangle-up', size=7, color='#006400') # Darker Green
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=pred_sells['ds'], y=pred_sells['high'] * 1.02,
    mode='markers', name='Predicted Sell',
    marker=dict(symbol='triangle-down', size=7, color='#8b0000') # Darker Red
), row=1, col=1)

# Row 2: Probabilities
fig.add_trace(go.Scatter(
    x=plot_df['ds'], y=plot_df['Prob_Up'],
    name="Prob Up (Buy)", line=dict(color='#00ff00', width=1)
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=plot_df['ds'], y=plot_df['Prob_Down'],
    name="Prob Down (Sell)", line=dict(color='#ff0000', width=1)
), row=2, col=1)

fig.update_layout(
    title=f'Backtest Analysis: {title_suffix}<br><sup>{stats_str}</sup>',
    template='plotly_dark',
    xaxis_rangeslider_visible=False,
    height=800
)

fig.show()
# %%