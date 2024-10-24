import argparse
from pathlib import Path
import torch
from torch.cuda.amp import autocast
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from config.config import NLNNConfig
from models.nlnn import NLNN
from training.trainer import NLNNTrainer
from utils.logging import get_logger
from utils.monitoring import ResourceMonitor
from utils.hardware_config import HardwareConfig, OptimizedDataPipeline, create_optimized_config

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Hardware-Optimized Normalized Liquid Neural Network")
    parser.add_argument('--config', type=str, default='config/config.json', help='Path to the config JSON file')
    parser.add_argument('--data', type=str, default='data/raw/btc_15m_data_2018_to_2024-2024-10-10.csv', help='Path to the training data CSV file')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs')
    return parser.parse_args()

def create_datasets(filepath: str, sequence_length: int, hw_config: HardwareConfig):
    df = pd.read_csv(filepath, parse_dates=['Open time'])
    df = df.sort_values('Open time')

    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[features].values

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i:i+sequence_length])
        y.append(data_scaled[i+sequence_length, 3])

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, shuffle=False)

    train_dataset = torch.utils.data.TensorDataset(
        hw_config.optimize_tensor(train_X),
        hw_config.optimize_tensor(train_y)
    )
    val_dataset = torch.utils.data.TensorDataset(
        hw_config.optimize_tensor(val_X),
        hw_config.optimize_tensor(val_y)
    )

    return train_dataset, val_dataset

def main():
    args = parse_args()
    logger = get_logger('n-LNN', log_dir=Path(args.log_dir))
    
    hw_config = create_optimized_config()
    logger.info(f"Using device: {hw_config.device}")
    logger.info(f"Batch size: {hw_config.batch_size} (effective: {hw_config.effective_batch_size})")
    
    config_path = Path(args.config)
    if config_path.exists():
        config = NLNNConfig.load(config_path)
    else:
        config = NLNNConfig()
        config.save(config_path)
    
    config.batch_size = hw_config.batch_size
    config.num_threads = hw_config.torch_threads
    config.use_mixed_precision = True
    
    logger.info("Creating datasets...")
    train_dataset, val_dataset = create_datasets(
        filepath=args.data,
        sequence_length=config.sequence_length,
        hw_config=hw_config
    )
    
    logger.info("Setting up data pipeline...")
    data_pipeline = OptimizedDataPipeline(hw_config)
    train_loader, val_loader = data_pipeline.create_dataloaders(
        train_dataset,
        val_dataset
    )
    
    logger.info("Initializing model...")
    model = NLNN(config)
    model = model.to(
        device=hw_config.device,
        memory_format=hw_config.memory_format
    )
    
    logger.info("Initializing trainer...")
    trainer = NLNNTrainer(
        model=model,
        config=config
    )
    
    logger.info(f"Starting training for {args.epochs} epochs...")
    trainer.train(
        train_dataloader=train_loader,
        num_epochs=args.epochs,
        val_dataloader=val_loader,
        save_dir=Path(args.save_dir)
    )
    
    logger.info(f"Final GPU memory stats: {hw_config.memory_stats()}")

if __name__ == "__main__":
    main()