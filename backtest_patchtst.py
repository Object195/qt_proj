#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
import importlib.util
import torch
import torch.nn as nn
import pandas as pd
from transformers import PatchTSTConfig, PatchTSTForClassification
import numpy as np
# from config import PIPELINE # This will be replaced by dynamic import
from backtest_visualizer import BacktestVisualizer

# --- Dynamic Config Loading ---
# 1. Define model paths and load the associated config
model_dir = 'training/models/patchtst_vatc'
model_path = model_dir # PatchTST uses a directory path
config_path = os.path.join(model_dir, 'config_copy.py')

if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config copy not found at {config_path}. "
                            "Please ensure a model has been trained and the config was saved.")

spec = importlib.util.spec_from_file_location("config_model", config_path)
config_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_model)
PIPELINE = config_model.PIPELINE
print(f"Loaded configuration from {config_path}")

# 2. Load Model and Data
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
    mask = (dates_pd >= train_start) & (dates_pd <= train_end)
    title_suffix = f"PatchTST - Training Set ({PIPELINE['train_start_date']} to {PIPELINE['train_end_date']})"
elif VIEW_MODE == 'test':
    mask = (dates_pd > train_end) & (dates_pd <= test_end)
    title_suffix = f"PatchTST - Test Set ({PIPELINE['train_end_date']} to {PIPELINE['fetch_end_date']})"
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

# 4. Candlestick Visualization
visualizer = BacktestVisualizer(results_df, df_prices)
visualizer.plot(title_suffix=title_suffix)
# %%