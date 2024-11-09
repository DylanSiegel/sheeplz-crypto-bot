# src/main.py

from pathlib import Path
import torch
from datetime import datetime
import sys
from typing import Tuple
import pandas as pd

from config.config import TradingConfig, TargetType, NormalizationType
from pipeline.pipeline import TradingDataPipeline
from pipeline.handlers import DirectionTargetCalculator
from utils.logging import custom_logger
from pipeline.metrics import PipelineMetrics
from pipeline.market_indicators import TechnicalIndicators, IndicatorConfig
from data.kline_preprocessor import KlinePreprocessor
from utils.system_monitor import SystemMonitor

def create_config() -> TradingConfig:
    """Create and initialize configuration"""
    return TradingConfig(
        # Data Parameters
        sequence_length=60,            # 1-hour lookback
        prediction_horizon=5,          # 5-minute prediction
        batch_size=4096,              # Optimized for 8GB VRAM
        
        # Hardware Optimization
        num_workers=6,                # Half of Ryzen 9 7900X cores
        pin_memory=True,
        
        # Processing Parameters
        outlier_std_threshold=4.0,
        volume_percentile_threshold=0.9995,
        use_robust_scaling=True,
        
        # Target Configuration
        target_type=TargetType.DIRECTION,
        target_normalization=NormalizationType.ROBUST
    )

def process_and_save_data(
    config: TradingConfig,
    input_path: Path,
    output_path: Path,
    metrics: PipelineMetrics
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process kline data and save results"""
    system_monitor = SystemMonitor()
    
    try:
        # Initialize preprocessor
        preprocessor = KlinePreprocessor(config)
        
        # Load and process data
        custom_logger.logger.info(f"Loading data from {input_path}")
        raw_data = pd.read_parquet(input_path) if input_path.suffix == '.parquet' else pd.read_csv(input_path)
        
        # Process features and targets
        custom_logger.logger.info("Processing kline data...")
        features, targets = preprocessor.process_klines(raw_data)
        
        # Log processing metrics
        current_metrics = system_monitor.get_metrics()
        metrics.update('processing_time', current_metrics.cpu_percent)
        metrics.update('memory_usage', current_metrics.memory_percent)
        
        custom_logger.log_metrics({
            'cpu_usage': current_metrics.cpu_percent,
            'memory_usage': current_metrics.memory_percent,
            'gpu_utilization': current_metrics.gpu_utilization,
            'data_samples': len(features)
        }, step=0)
        
        # Save processed data
        custom_logger.logger.info(f"Saving processed data to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'features': features,
            'targets': targets,
            'config': config.to_dict(),
            'processing_timestamp': datetime.now().isoformat()
        }, output_path)
        
        # Log summary statistics
        custom_logger.logger.info(f"Processed {len(features):,} samples")
        custom_logger.logger.info(f"Feature shape: {features.shape}")
        custom_logger.logger.info(f"Target shape: {targets.shape}")
        
        return features, targets
        
    except Exception as e:
        custom_logger.logger.error(f"Error processing data: {str(e)}", exception=True)
        raise

def main():
    try:
        # Initialize configuration
        config = create_config()
        custom_logger.logger.info("Configuration initialized")
        
        # Initialize metrics tracker
        metrics = PipelineMetrics()
        
        # Define paths
        input_path = Path("src/data/raw/btc_usdt_1m_klines.parquet")
        output_path = Path("src/data/processed/kline_features.pt")
        
        # Process and save data
        features, targets = process_and_save_data(
            config,
            input_path,
            output_path,
            metrics
        )
        
        # Log final metrics
        metrics.log_summary()
        custom_logger.logger.info("Data processing completed successfully")
        
        # Optional: Create training/validation splits
        if features is not None and targets is not None:
            split_idx = int(len(features) * 0.8)
            
            split_data = {
                'train_features': features[:split_idx],
                'train_targets': targets[:split_idx],
                'val_features': features[split_idx:],
                'val_targets': targets[split_idx:],
                'config': config.to_dict()
            }
            
            split_path = output_path.parent / 'train_val_split.pt'
            torch.save(split_data, split_path)
            
            custom_logger.logger.info(
                f"Train/validation split saved - "
                f"Train: {len(split_data['train_features']):,} samples, "
                f"Val: {len(split_data['val_features']):,} samples"
            )
        
    except Exception as e:
        custom_logger.logger.error(f"Pipeline failed: {str(e)}", exception=True)
        sys.exit(1)

if __name__ == "__main__":
    main()