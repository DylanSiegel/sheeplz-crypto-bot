# File: lnn_module.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, cpu_count  # Correctly import Pool and cpu_count
from torch.cuda.amp import GradScaler, autocast

# Define file paths for all timeframes
PROCESSED_DATA_DIR = r"data\final"
PROCESSED_DATA_PATHS = {
    "15m": os.path.join(PROCESSED_DATA_DIR, "processed_data_15m.csv.gz"),
    "1h": os.path.join(PROCESSED_DATA_DIR, "processed_data_1h.csv.gz"),
    "4h": os.path.join(PROCESSED_DATA_DIR, "processed_data_4h.csv.gz"),
    "1d": os.path.join(PROCESSED_DATA_DIR, "processed_data_1d.csv.gz")
}

# Parameters
TARGET_COLUMNS = {
    '15m': ['target_15m', 'target_1h', 'target_4h', 'target_1d'],
    '1h': ['target_1h', 'target_4h', 'target_1d'],
    '4h': ['target_4h', 'target_1d'],
    '1d': ['target_1d']
}
FEATURE_COLUMNS = {
    '15m': [
        'open', 'high', 'low', 'close', 'volume',
        'return_15m', 'return_1h', 'return_4h', 'return_1d',
        'ema_14', 'sma_14', 'rsi_14', 'stoch_k', 'stoch_d',
        'bb_mavg', 'bb_hband', 'bb_lband', 'bb_pband', 'bb_wband',
        'kc_hband', 'kc_lband', 'kc_mband', 'kc_pband', 'kc_wband',
        'atr_14', 'obv', 'macd', 'macd_signal', 'macd_diff',
        'adx', 'adx_pos', 'adx_neg', 'ulcer_index',
        'adi', 'cmf', 'eom', 'vpt'
    ],
    '1h': [
        'open', 'high', 'low', 'close', 'volume',
        'return_15m', 'return_1h', 'return_4h', 'return_1d',
        'ema_14', 'sma_14', 'rsi_14', 'stoch_k', 'stoch_d',
        'bb_mavg', 'bb_hband', 'bb_lband', 'bb_pband', 'bb_wband',
        'kc_hband', 'kc_lband', 'kc_mband', 'kc_pband', 'kc_wband',
        'atr_14', 'obv', 'macd', 'macd_signal', 'macd_diff',
        'adx', 'adx_pos', 'adx_neg', 'ulcer_index',
        'adi', 'cmf', 'eom', 'vpt'
    ],
    '4h': [
        'open', 'high', 'low', 'close', 'volume',
        'return_15m', 'return_1h', 'return_4h', 'return_1d',
        'ema_14', 'sma_14', 'rsi_14', 'stoch_k', 'stoch_d',
        'bb_mavg', 'bb_hband', 'bb_lband', 'bb_pband', 'bb_wband',
        'kc_hband', 'kc_lband', 'kc_mband', 'kc_pband', 'kc_wband',
        'atr_14', 'obv', 'macd', 'macd_signal', 'macd_diff',
        'adx', 'adx_pos', 'adx_neg', 'ulcer_index',
        'adi', 'cmf', 'eom', 'vpt'
    ],
    '1d': [
        'open', 'high', 'low', 'close', 'volume',
        'return_15m', 'return_1h', 'return_4h', 'return_1d',
        'ema_14', 'sma_14', 'rsi_14', 'stoch_k', 'stoch_d',
        'bb_mavg', 'bb_hband', 'bb_lband', 'bb_pband', 'bb_wband',
        'kc_hband', 'kc_lband', 'kc_mband', 'kc_pband', 'kc_wband',
        'atr_14', 'obv', 'macd', 'macd_signal', 'macd_diff',
        'adx', 'adx_pos', 'adx_neg', 'ulcer_index',
        'adi', 'cmf', 'eom', 'vpt'
    ]
}
SEQ_LENGTH = 10
BATCH_SIZE = 256  # Adjust based on GPU memory
NUM_WORKERS = cpu_count()  # Define NUM_WORKERS using cpu_count()
SCALER_DIR = 'scalers'
MODEL_DIR = 'models'
NUM_EPOCHS = 50
LEARNING_RATE = 0.0001

# Ensure directories exist
os.makedirs(SCALER_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Enable CUDA benchmark for optimized performance on fixed input sizes
torch.backends.cudnn.benchmark = True

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_preprocess(processed_path, feature_cols, target_cols, scaler_path):
    """
    Loads and preprocesses the data.

    Parameters:
    - processed_path: Path to the processed CSV file.
    - feature_cols: List of feature column names.
    - target_cols: List of target column names.
    - scaler_path: Path to save/load the scaler.

    Returns:
    - df_scaled: DataFrame with scaled features and targets.
    """
    # Load processed data
    df_processed = pd.read_csv(
        processed_path,
        parse_dates=['open time'],
        dtype={col: np.float32 for col in feature_cols},
        compression='gzip'
    )

    # Handle missing values using forward and backward fill
    df_processed.ffill(inplace=True)
    df_processed.bfill(inplace=True)

    # Feature scaling
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_processed[feature_cols].astype(np.float32))
    df_scaled = pd.DataFrame(scaled_features, columns=feature_cols, index=df_processed.index)
    df_scaled['open time'] = df_processed['open time']
    for target in target_cols:
        df_scaled[target] = df_processed[target].astype(np.float32)

    # Save the scaler
    joblib.dump(scaler, scaler_path)

    return df_scaled

def create_sequences(data, seq_length, feature_cols, target_cols):
    """
    Creates input sequences and corresponding targets.

    Parameters:
    - data: DataFrame containing features and targets.
    - seq_length: Number of time steps in each input sequence.
    - feature_cols: List of feature column names.
    - target_cols: List of target column names.

    Returns:
    - sequences: Numpy array of shape (num_samples, seq_length, num_features)
    - targets: Dictionary of Numpy arrays for each target
    """
    sequences = []
    targets = {target: [] for target in target_cols}

    for i in range(len(data) - seq_length):
        seq = data[feature_cols].iloc[i:i+seq_length].values
        sequences.append(seq)
        for target in target_cols:
            targets[target].append(data[target].iloc[i+seq_length])

    sequences = np.array(sequences, dtype=np.float32)
    for target in target_cols:
        targets[target] = np.array(targets[target], dtype=np.float32)

    return sequences, targets

class FinancialDataset(Dataset):
    """
    Custom Dataset for financial data sequences.
    """
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = {key: torch.tensor(val, dtype=torch.float32).unsqueeze(1) for key, val in targets.items()}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        target = {key: self.targets[key][idx] for key in self.targets}
        return sequence, target

class LNNModel(nn.Module):
    """
    Optimized Liquid Neural Network Model using LSTM layers with dropout and batch normalization.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.3):
        super(LNNModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # Forward propagate LSTM
        out, _ = self.lstm(x)

        # Apply batch normalization on the last time step
        out = self.batch_norm(out[:, -1, :])

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out

def train_model(timeframe):
    """
    Trains the LNN model for a specific timeframe.

    Parameters:
    - timeframe: Timeframe identifier (e.g., '15m', '1h').

    Returns:
    - None
    """
    processed_path = PROCESSED_DATA_PATHS[timeframe]
    feature_cols = FEATURE_COLUMNS[timeframe]
    target_cols = TARGET_COLUMNS[timeframe]
    scaler_path = os.path.join(SCALER_DIR, f"scaler_{timeframe}.save")
    model_save_path = os.path.join(MODEL_DIR, f"lnn_model_{timeframe}.pth")

    # Load and preprocess data
    df_scaled = load_and_preprocess(processed_path, feature_cols, target_cols, scaler_path)

    # Create sequences and targets
    sequences, targets = create_sequences(df_scaled, SEQ_LENGTH, feature_cols, target_cols)

    # For simplicity, let's train to predict the first target in target_cols
    primary_target = target_cols[0]
    dataset = FinancialDataset(sequences, {primary_target: targets[primary_target]})
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # Initialize model, loss function, and optimizer
    input_size = len(feature_cols)
    hidden_size = 256
    num_layers = 3
    output_size = 1

    model = LNNModel(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    # Initialize GradScaler for mixed precision training
    scaler = GradScaler()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        progress = tqdm(
            data_loader,
            desc=f"{timeframe} Epoch {epoch+1}/{NUM_EPOCHS}",
            leave=False,
            ncols=100
        )

        for sequences_batch, targets_batch in progress:
            sequences_batch = sequences_batch.to(device, non_blocking=True)
            targets_batch = targets_batch.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Mixed precision forward pass
            with autocast():
                outputs = model(sequences_batch)
                loss = criterion(outputs, targets_batch)

            # Scales loss and backpropagates
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(data_loader)
        print(f"{timeframe} Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.6f}")

        # Step the scheduler
        scheduler.step(avg_loss)

    # Save the trained model
    torch.save(model.state_dict(), model_save_path)
    print(f"{timeframe} Model training complete and saved to {model_save_path}")

def main():
    """
    Main function to train models for all timeframes in parallel.
    """
    timeframes = list(PROCESSED_DATA_PATHS.keys())
    with Pool(processes=cpu_count()) as pool:
        pool.map(train_model, timeframes)

if __name__ == "__main__":
    main()
