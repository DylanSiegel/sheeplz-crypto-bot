import argparse
import sys
from pathlib import Path
from loguru import logger
import torch
from typing import Dict, List, Optional, Tuple
import json

class ProcessorConfig:
    """Configuration for data processing"""
    def __init__(
        self,
        raw_dir: Path = Path("data/raw"),
        processed_dir: Path = Path("data/processed"),
        timeframes: List[str] = ['15m', '1h', '4h', '1d'],
        sequence_length: int = 128,
        batch_size: int = 512,
        train_ratio: float = 0.8,
        num_threads: int = 12,
        use_cuda: bool = torch.cuda.is_available()
    ):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.timeframes = timeframes
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.num_threads = num_threads
        self.use_cuda = use_cuda

def setup_logger():
    """Configure logging"""
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(
        "logs/data_processor.log",
        rotation="500 MB",
        retention="10 days"
    )

def process_single_timeframe(processor, timeframe: str, config: ProcessorConfig) -> Dict:
    """Process a single timeframe and return stats"""
    try:
        logger.info(f"Processing timeframe: {timeframe}")
        
        start_time = time.time()
        X_train, y_train, X_val, y_val = processor.process_timeframe_distributed(timeframe)
        processing_time = time.time() - start_time
        
        stats = {
            'timeframe': timeframe,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'feature_dim': X_train.shape[-1],
            'sequence_length': X_train.shape[1],
            'processing_time': processing_time,
            'memory_used': processor.get_memory_stats()
        }
        
        logger.info(f"Processed {timeframe} in {processing_time:.2f}s")
        logger.info(f"Train samples: {stats['train_samples']}, Val samples: {stats['val_samples']}")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error processing {timeframe}: {str(e)}")
        raise

def process_all_timeframes(processor, config: ProcessorConfig) -> List[Dict]:
    """Process all timeframes and return stats"""
    stats = []
    
    for timeframe in config.timeframes:
        try:
            timeframe_stats = process_single_timeframe(processor, timeframe, config)
            stats.append(timeframe_stats)
        except Exception as e:
            logger.error(f"Skipping {timeframe} due to error: {str(e)}")
            continue
            
    return stats

def save_stats(stats: List[Dict], output_dir: Path):
    """Save processing statistics"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "processing_stats.json"
    
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved processing stats to {output_file}")

def create_data_visualizations(stats: List[Dict], output_dir: Path):
    """Create visualizations of the processed data"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set up the plotting style
        plt.style.use('seaborn')
        
        # Processing time comparison
        plt.figure(figsize=(10, 6))
        times = [stat['processing_time'] for stat in stats]
        timeframes = [stat['timeframe'] for stat in stats]
        plt.bar(timeframes, times)
        plt.title('Processing Time by Timeframe')
        plt.xlabel('Timeframe')
        plt.ylabel('Time (seconds)')
        plt.savefig(output_dir / 'processing_times.png')
        plt.close()
        
        # Sample distribution
        plt.figure(figsize=(10, 6))
        train_samples = [stat['train_samples'] for stat in stats]
        val_samples = [stat['val_samples'] for stat in stats]
        x = np.arange(len(timeframes))
        width = 0.35
        
        plt.bar(x - width/2, train_samples, width, label='Train')
        plt.bar(x + width/2, val_samples, width, label='Validation')
        plt.xticks(x, timeframes)
        plt.title('Sample Distribution by Timeframe')
        plt.xlabel('Timeframe')
        plt.ylabel('Number of Samples')
        plt.legend()
        plt.savefig(output_dir / 'sample_distribution.png')
        plt.close()
        
        logger.info(f"Saved visualizations to {output_dir}")
        
    except Exception as e:
        logger.warning(f"Could not create visualizations: {str(e)}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Process cryptocurrency data across multiple timeframes')
    parser.add_argument('--raw-dir', type=str, default='data/raw', help='Directory containing raw data files')
    parser.add_argument('--processed-dir', type=str, default='data/processed', help='Directory for processed data')
    parser.add_argument('--timeframes', type=str, nargs='+', default=['15m', '1h', '4h', '1d'], help='Timeframes to process')
    parser.add_argument('--sequence-length', type=int, default=128, help='Sequence length for time series')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size for data loading')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Train/validation split ratio')
    parser.add_argument('--num-threads', type=int, default=12, help='Number of processing threads')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Directory for outputs and visualizations')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logger()
    if args.debug:
        logger.level("DEBUG")
    
    # Create configuration
    config = ProcessorConfig(
        raw_dir=Path(args.raw_dir),
        processed_dir=Path(args.processed_dir),
        timeframes=args.timeframes,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_threads=args.num_threads
    )
    
    # Initialize processor
    try:
        logger.info("Initializing data processor...")
        from enhanced_processor import EnhancedDataProcessor  # Import your processor class
        processor = EnhancedDataProcessor(config)
        
        # Process timeframes
        logger.info("Starting data processing...")
        stats = process_all_timeframes(processor, config)
        
        # Save results
        output_dir = Path(args.output_dir)
        save_stats(stats, output_dir)
        create_data_visualizations(stats, output_dir)
        
        logger.info("Data processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        if args.debug:
            import traceback
            logger.debug(traceback.format_exc())
        sys.exit(1)
    finally:
        if 'processor' in locals():
            processor.cleanup()

if __name__ == "__main__":
    main()