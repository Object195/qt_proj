#%%
import sys
import os
import shutil
from datetime import datetime
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Add the project root to sys.path to allow importing config.py from the parent directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import PatchTSTConfig, PatchTSTForClassification
from tqdm import tqdm
import plotly.graph_objects as go
from config import PIPELINE, PATCHTST_PARAMS

# 1. Load Data
data_path = os.path.join(PROJECT_ROOT, 'training', 'datasets', 'train_data.pt')
if not os.path.exists(data_path):
    raise FileNotFoundError(f"{data_path} not found. Run generate_data.py first.")

train_data = torch.load(data_path, weights_only=False)
X_train = train_data['X'] # Shape: (N, Window, Features)
y_train = train_data['y'] # Shape: (N,)

print(f"Training Data Shape: {X_train.shape}")

# --- LABEL VALIDATION ---
unique_labels = torch.unique(y_train)
num_classes = PATCHTST_PARAMS['num_classes']
print(f"Unique labels in dataset: {unique_labels.tolist()}")

if y_train.min() < 0 or y_train.max() >= num_classes:
    raise ValueError(f"Label mismatch! Found labels in range [{y_train.min()}, {y_train.max()}], "
                     f"but num_classes is set to {num_classes}. Labels must be in [0, {num_classes-1}].")

# 2. Determine Dynamic Parameters
num_samples, context_length, num_features = X_train.shape

# 3. Configure PatchTST
config = PatchTSTConfig(
    num_input_channels=num_features,      # Determined from data
    context_length=context_length,        # Window size (30)
    prediction_length=PIPELINE['forecast_horizon'], # Required by config, though unused for classification head
    patch_length=PATCHTST_PARAMS['patch_length'],
    stride=PATCHTST_PARAMS['stride'],
    num_hidden_layers=PATCHTST_PARAMS['num_hidden_layers'],
    num_attention_heads=PATCHTST_PARAMS['num_attention_heads'],
    num_classes=PATCHTST_PARAMS['num_classes'],
    num_labels=PATCHTST_PARAMS['num_classes'], # Explicitly set num_labels for HF compatibility
    dropout=PATCHTST_PARAMS['dropout'],
    use_cls_token=True # Use CLS token for classification
)

model = PatchTSTForClassification(config)

# --- FIX: Ensure Head Output Dimension ---
# Run a dummy forward pass to check output shape and fix head if necessary
print("Verifying model output shape...")
with torch.no_grad():
    # Create dummy input on CPU
    dummy_input = torch.randn(2, context_length, num_features)
    dummy_out = model(past_values=dummy_input).prediction_logits
    print(f"Initial model output shape: {dummy_out.shape}")
    
    # We check if the last dimension is wrong OR if the tensor has too many dimensions (> 2D)
    if dummy_out.shape[-1] != PATCHTST_PARAMS['num_classes'] or dummy_out.dim() > 2:
        print(f"Shape mismatch! Expected 2D tensor (Batch, {PATCHTST_PARAMS['num_classes']}), got {list(dummy_out.shape)}. Re-initializing head.")
        
        # Update config to ensure correct saving/loading
        model.config.num_classes = PATCHTST_PARAMS['num_classes']
        model.config.num_labels = PATCHTST_PARAMS['num_classes']
        
        # 1. Define a custom pooling layer to flatten the extra dimensions
        class GlobalAvgPool(nn.Module):
            def forward(self, x):
                # Input x shape: (batch_size, num_channels, num_patches, d_model)
                # Output shape: (batch_size, d_model) by taking the mean over channels and patches
                return x.mean(dim=(1, 2))
                
        # 2. Replace the head with the Pooling layer included
        model.head = nn.Sequential(
            GlobalAvgPool(),
            nn.Dropout(model.config.dropout),
            nn.Linear(model.config.d_model, PATCHTST_PARAMS['num_classes'])
        )

# 4. Setup Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=PATCHTST_PARAMS['batch_size'], shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=PATCHTST_PARAMS['learning_rate'])
criterion = nn.CrossEntropyLoss()

# 5. Training Loop
print(f"Starting training on {device}...")
model.train()
epoch_losses = []

progress_bar = tqdm(range(PATCHTST_PARAMS['epochs']), desc="Training Progress")
for epoch in progress_bar:
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        
        # PatchTST expects 'past_values' for the input time series
        outputs = model(past_values=batch_X)
        logits = outputs.prediction_logits
        
        # Ensure logits are (Batch, NumClasses) - Squeeze sequence dim if present
        if logits.dim() == 3 and logits.shape[1] == 1:
            logits = logits.squeeze(1)

        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)
        
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    epoch_losses.append(avg_loss)
    progress_bar.set_postfix({'loss': f"{avg_loss:.4f}", 'acc': f"{accuracy:.4f}"})

# 6. Plot Convergence
fig = go.Figure()
fig.add_trace(go.Scatter(y=epoch_losses, mode='lines+markers', name='Training Loss'))
fig.update_layout(
    title='PatchTST Training Convergence',
    xaxis_title='Epoch',
    yaxis_title='Cross Entropy Loss',
    template='plotly_dark'
)
fig.show()

# 7. Save Model
model_save_dir = os.path.join(PROJECT_ROOT, 'training', 'models', 'patchtst_vatc')
model.save_pretrained(model_save_dir)
print(f"Model saved to {model_save_dir}")

# Save a copy of the config file
if PIPELINE.get('save_config_with_timestamp', False):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_filename = f'config_{timestamp}.py'
else:
    config_filename = 'config_copy.py'

config_src = os.path.join(PROJECT_ROOT, 'config.py')
config_dst = os.path.join(model_save_dir, config_filename)
shutil.copy2(config_src, config_dst)
print(f"Config saved to {config_dst}")