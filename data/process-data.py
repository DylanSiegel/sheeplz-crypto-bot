import pandas as pd
import numpy as np
from numba import jit
from pathlib import Path
import torch
from typing import Tuple, Dict, List, NamedTuple, Optional
import ta
from sklearn.preprocessing import RobustScaler
from loguru import logger
import ray
from tqdm import tqdm
import psutil
import time
import json
from datetime import datetime
import warnings
import gc
import sys
from dataclasses import dataclass, asdict

@dataclass
class ProcessingStats:
    """Statistics for data processing"""
    timeframe: str
    start_time: str
    end_time: str
    total_rows: int
    processed_rows: int
    failed_chunks: int
    memory_usage: Dict[str, float]
    processing_time: float
    chunk_times: List[float]
    feature_stats: Dict[str, Dict[str, float]]

class ChunkData(NamedTuple):
    """Serializable container for chunk data"""
    timestamps: np.ndarray
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    volumes: np.ndarray

def validate_ohlcv_data(
    df: pd.DataFrame,
    timeframe: str
) -> Tuple[bool, List[str]]:
    """Validate OHLCV data quality"""
    issues = []
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        issues.append(f"Missing values found: {missing[missing > 0].to_dict()}")
    
    # Check for negative values
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if (df[col] <= 0).any():
            issues.append(f"Negative or zero values found in {col}")
    
    # Check price consistency
    invalid_prices = (
        (df['High'] < df['Low']) |
        (df['Open'] > df['High']) |
        (df['Open'] < df['Low']) |
        (df['Close'] > df['High']) |
        (df['Close'] < df['Low'])
    )
    if invalid_prices.any():
        issues.append(f"Found {invalid_prices.sum()} rows with invalid price relationships")
    
    # Check timestamp consistency
    expected_diff = pd.Timedelta(timeframe)
    time_diffs = df['Open time'].diff()
    invalid_times = time_diffs != expected_diff
    if invalid_times.sum() > 1:  # Allow for one difference due to first row
        issues.append(f"Found {invalid_times.sum()-1} unexpected timestamp intervals")
    
    return len(issues) == 0, issues

@jit(nopython=True)
def calculate_features(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray
) -> Dict[str, np.ndarray]:
    """Calculate basic features using Numba"""
    length = len(closes)
    
    # Initialize arrays
    returns = np.zeros(length)
    volatility = np.zeros(length)
    volume_intensity = np.zeros(length)
    log_returns = np.zeros(length)
    
    # Calculate features
    for i in range(1, length):
        returns[i] = closes[i] / closes[i-1] - 1
        log_returns[i] = np.log(closes[i] / closes[i-1])
        
        price_range = highs[i] - lows[i]
        if price_range > 0:
            volume_intensity[i] = volumes[i] * (price_range / opens[i])
        
        if opens[i] != 0 and closes[i] != 0:
            volatility[i] = np.sqrt(
                0.5 * np.log(highs[i]/lows[i])**2 -
                (2*np.log(2)-1) * np.log(closes[i]/opens[i])**2
            )
    
    return {
        'returns': returns,
        'log_returns': log_returns,
        'volatility': volatility,
        'volume_intensity': volume_intensity
    }

@ray.remote
def process_chunk(chunk_data: ChunkData) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, float]]]:
    """Process a single chunk with feature calculation and statistics"""
    start_time = time.time()
    
    # Convert timestamps back to datetime
    timestamps = pd.to_datetime(chunk_data.timestamps)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Open time': timestamps,
        'Open': chunk_data.opens,
        'High': chunk_data.highs,
        'Low': chunk_data.lows,
        'Close': chunk_data.closes,
        'Volume': chunk_data.volumes
    })
    
    # Calculate base features
    base_features = calculate_features(
        chunk_data.opens,
        chunk_data.highs,
        chunk_data.lows,
        chunk_data.closes,
        chunk_data.volumes
    )
    
    features = base_features.copy()
    
    try:
        # Volume indicators
        features['vwap'] = ta.volume.volume_weighted_average_price(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            volume=df['Volume']
        ).values
        
        # Trend indicators
        features['ema_12'] = ta.trend.ema_indicator(close=df['Close'], window=12).values
        features['ema_26'] = ta.trend.ema_indicator(close=df['Close'], window=26).values
        features['macd'] = ta.trend.macd_diff(close=df['Close']).values
        
        # Momentum indicators
        features['rsi'] = ta.momentum.rsi(close=df['Close']).values
        features['stoch_k'] = ta.momentum.stoch(
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        ).values
        
        # Volatility indicators
        bb_high = ta.volatility.bollinger_hband(close=df['Close'])
        bb_low = ta.volatility.bollinger_lband(close=df['Close'])
        features['bb_width'] = ((bb_high - bb_low) / df['Close']).values
        features['atr'] = ta.volatility.average_true_range(
            high=df['High'],
            low=df['Low'],
            close=df['Close']
        ).values
        
    except Exception as e:
        logger.warning(f"Error calculating technical indicators: {e}")
    
    # Calculate statistics for each feature
    feature_stats = {}
    for name, values in features.items():
        clean_values = values[~np.isnan(values)]
        if len(clean_values) > 0:
            feature_stats[name] = {
                'mean': float(np.mean(clean_values)),
                'std': float(np.std(clean_values)),
                'min': float(np.min(clean_values)),
                'max': float(np.max(clean_values)),
                'null_count': int(np.isnan(values).sum())
            }
    
    # Replace NaN values
    for key in features:
        features[key] = np.nan_to_num(features[key], nan=0.0, posinf=0.0, neginf=0.0)
    
    return features, feature_stats

class CryptoDataProcessor:
    def __init__(self, data_dir: str = 'data/raw', output_dir: str = 'data/processed'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self._setup_logging()
        self._init_ray()
        self.stats = {}
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_path = self.output_dir / 'logs'
        log_path.mkdir(parents=True, exist_ok=True)
        
        logger.remove()
        logger.add(
            log_path / "crypto_processing_{time}.log",
            rotation="1 GB",
            compression="zip",
            level="INFO",
            backtrace=True,
            diagnose=True
        )
        
        warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    def _init_ray(self):
        """Initialize Ray with proper configuration"""
        if not ray.is_initialized():
            ray.init(
                num_cpus=12,
                num_gpus=1,
                include_dashboard=True,
                ignore_reinit_error=True,
                logging_level=logging.INFO,
                _memory=8 * 1024 * 1024 * 1024  # 8GB memory limit
            )
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss': memory_info.rss / 1024**2,  # RSS in MB
            'vms': memory_info.vms / 1024**2,  # VMS in MB
            'percent': process.memory_percent(),
            'available': psutil.virtual_memory().available / 1024**2  # Available memory in MB
        }
    
    def process_timeframe(
        self,
        timeframe: str,
        sequence_length: int = 128,
        train_ratio: float = 0.8,
        chunk_size: int = 50000,
        max_retries: int = 3
    ) -> Dict[str, torch.Tensor]:
        """Process a single timeframe with monitoring and error recovery"""
        logger.info(f"Starting processing of {timeframe} timeframe")
        start_time = datetime.now()
        
        # Initialize statistics
        stats = ProcessingStats(
            timeframe=timeframe,
            start_time=start_time.isoformat(),
            end_time='',
            total_rows=0,
            processed_rows=0,
            failed_chunks=0,
            memory_usage={},
            processing_time=0.0,
            chunk_times=[],
            feature_stats={}
        )
        
        try:
            # Load and validate data
            df = self._load_timeframe_data(timeframe)
            stats.total_rows = len(df)
            
            is_valid, issues = validate_ohlcv_data(df, timeframe)
            if not is_valid:
                for issue in issues:
                    logger.warning(f"Data validation issue: {issue}")
            
            # Process in chunks with retry mechanism
            futures = []
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i + chunk_size].copy()
                retries = 0
                
                while retries < max_retries:
                    try:
                        chunk_data = self._create_chunk_data(chunk)
                        futures.append(process_chunk.remote(chunk_data))
                        break
                    except Exception as e:
                        logger.error(f"Error processing chunk (attempt {retries + 1}/{max_retries}): {e}")
                        retries += 1
                        if retries == max_retries:
                            stats.failed_chunks += 1
                
                # Memory management
                del chunk
                gc.collect()
            
            # Collect results with monitoring
            processed_features = []
            all_feature_stats = []
            
            for future in tqdm(futures, desc=f"Processing {timeframe} chunks"):
                try:
                    chunk_start = time.time()
                    features, feature_stats = ray.get(future)
                    chunk_time = time.time() - chunk_start
                    
                    processed_features.append(features)
                    all_feature_stats.append(feature_stats)
                    stats.chunk_times.append(chunk_time)
                    stats.processed_rows += chunk_size
                    
                except Exception as e:
                    logger.error(f"Error collecting chunk results: {e}")
                    stats.failed_chunks += 1
            
            # Combine features and calculate overall statistics
            combined_features = {}
            for key in processed_features[0].keys():
                combined_features[key] = np.concatenate([
                    chunk[key] for chunk in processed_features
                ])
            
            # Calculate final feature statistics
            stats.feature_stats = {}
            for feature_name in combined_features.keys():
                stats.feature_stats[feature_name] = {
                    'mean': float(np.mean([fs[feature_name]['mean'] for fs in all_feature_stats])),
                    'std': float(np.mean([fs[feature_name]['std'] for fs in all_feature_stats])),
                    'min': float(min([fs[feature_name]['min'] for fs in all_feature_stats])),
                    'max': float(max([fs[feature_name]['max'] for fs in all_feature_stats])),
                    'null_percentage': float(sum([fs[feature_name]['null_count'] for fs in all_feature_stats]) / stats.total_rows * 100)
                }
            
            # Create feature matrix and normalize
            feature_names = sorted(combined_features.keys())
            feature_matrix = np.column_stack([combined_features[name] for name in feature_names])
            
            scaler = RobustScaler()
            features_scaled = scaler.fit_transform(feature_matrix)
            
            # Generate labels and prepare sequences
            labels = self._generate_labels(combined_features)
            sequences = self._prepare_sequences(
                features_scaled, labels, sequence_length, train_ratio
            )
            
            # Update final statistics
            end_time = datetime.now()
            stats.end_time = end_time.isoformat()
            stats.processing_time = (end_time - start_time).total_seconds()
            stats.memory_usage = self._get_memory_usage()
            
            # Save statistics
            self._save_statistics(stats)
            
            return dict(zip(['X_train', 'y_train', 'X_val', 'y_val'], sequences))
            
        except Exception as e:
            logger.error(f"Error processing timeframe {timeframe}: {e}")
            raise
    
    def _save_statistics(self, stats: ProcessingStats):
        """Save processing statistics to file"""
        stats_path = self.output_dir / 'statistics'
        stats_path.mkdir(parents=True, exist_ok=True)
        
        stats_file = stats_path / f"{stats.timeframe}_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(asdict(stats), f, indent=2)
    
    def process_all_timeframes(
        self,
        sequence_length: int = 128,
        train_ratio: float = 0.8
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Process all timeframes with comprehensive monitoring"""
        timeframes = ['15m', '1h', '4h', '1d']
        processed_data = {}
        
        overall_start = time.time()
        total_memory_start = psutil.Process().memory_info().rss
        
        try:
            for timeframe in timeframes:
                logger.info(f"\nProcessing {timeframe} timeframe...")
                processed_data[timeframe] = self.process_timeframe(
                    timeframe,
                    sequence_length,
                    train_ratio
                )
        
        finally:# Continue from previous implementation...
            
            total_memory_end = psutil.Process().memory_info().rss
            overall_time = time.time() - overall_start
            
            # Save overall processing statistics
            overall_stats = {
                'total_processing_time': overall_time,
                'memory_delta_mb': (total_memory_end - total_memory_start) / (1024 * 1024),
                'timeframes_processed': len(timeframes),
                'total_samples_processed': sum(self.stats[tf].processed_rows for tf in timeframes if tf in self.stats),
                'failed_chunks_total': sum(self.stats[tf].failed_chunks for tf in timeframes if tf in self.stats)
            }
            
            with open(self.output_dir / 'overall_stats.json', 'w') as f:
                json.dump(overall_stats, f, indent=2)
            
            return processed_data
    
    def save_processed_data(
        self,
        processed_data: Dict[str, Dict[str, torch.Tensor]],
        compress: bool = True
    ):
        """Save processed data with optional compression"""
        logger.info("Saving processed data...")
        
        for timeframe, data in processed_data.items():
            timeframe_dir = self.output_dir / timeframe
            timeframe_dir.mkdir(parents=True, exist_ok=True)
            
            # Save data tensors
            for name, tensor in data.items():
                filepath = timeframe_dir / f"{name}.pt"
                try:
                    if compress:
                        # Use PyTorch's built-in compression
                        torch.save(
                            tensor,
                            filepath,
                            _use_new_zipfile_serialization=True
                        )
                    else:
                        torch.save(tensor, filepath)
                    
                    logger.info(f"Saved {timeframe} {name} to {filepath}")
                    
                except Exception as e:
                    logger.error(f"Error saving {timeframe} {name}: {e}")
            
            # Save tensor metadata
            metadata = {
                name: {
                    'shape': tensor.shape,
                    'dtype': str(tensor.dtype),
                    'device': str(tensor.device),
                    'size_mb': tensor.element_size() * tensor.nelement() / (1024 * 1024)
                }
                for name, tensor in data.items()
            }
            
            with open(timeframe_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def generate_report(self):
        """Generate comprehensive processing report"""
        report = {
            'processing_summary': {
                'total_timeframes': len(self.stats),
                'total_processing_time': sum(s.processing_time for s in self.stats.values()),
                'total_rows_processed': sum(s.processed_rows for s in self.stats.values()),
                'total_failed_chunks': sum(s.failed_chunks for s in self.stats.values())
            },
            'timeframe_details': {
                timeframe: {
                    'processing_time': stats.processing_time,
                    'rows_processed': stats.processed_rows,
                    'failed_chunks': stats.failed_chunks,
                    'average_chunk_time': np.mean(stats.chunk_times),
                    'memory_usage': stats.memory_usage
                }
                for timeframe, stats in self.stats.items()
            },
            'feature_analysis': {
                timeframe: {
                    'feature_stats': stats.feature_stats
                }
                for timeframe, stats in self.stats.items()
            }
        }
        
        # Save report
        report_path = self.output_dir / 'processing_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate HTML report
        html_report = self._generate_html_report(report)
        with open(self.output_dir / 'processing_report.html', 'w') as f:
            f.write(html_report)
    
    def _generate_html_report(self, report: Dict) -> str:
        """Generate HTML report from processing statistics"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Crypto Data Processing Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin-bottom: 30px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .warning { color: orange; }
                .error { color: red; }
            </style>
        </head>
        <body>
        """
        
        # Add summary section
        html += "<h1>Crypto Data Processing Report</h1>"
        html += "<div class='section'>"
        html += "<h2>Processing Summary</h2>"
        html += "<table>"
        for key, value in report['processing_summary'].items():
            html += f"<tr><th>{key}</th><td>{value}</td></tr>"
        html += "</table></div>"
        
        # Add timeframe details
        html += "<div class='section'>"
        html += "<h2>Timeframe Details</h2>"
        html += "<table><tr><th>Timeframe</th><th>Processing Time</th><th>Rows Processed</th>"
        html += "<th>Failed Chunks</th><th>Avg Chunk Time</th><th>Memory Usage</th></tr>"
        
        for timeframe, details in report['timeframe_details'].items():
            html += f"""
            <tr>
                <td>{timeframe}</td>
                <td>{details['processing_time']:.2f}s</td>
                <td>{details['rows_processed']}</td>
                <td class='{"error" if details["failed_chunks"] > 0 else ""}'>{details['failed_chunks']}</td>
                <td>{details['average_chunk_time']:.2f}s</td>
                <td>{details['memory_usage']['rss']:.1f}MB</td>
            </tr>
            """
        html += "</table></div>"
        
        # Add feature analysis
        html += "<div class='section'>"
        html += "<h2>Feature Analysis</h2>"
        
        for timeframe, analysis in report['feature_analysis'].items():
            html += f"<h3>Timeframe: {timeframe}</h3>"
            html += "<table><tr><th>Feature</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th><th>Null %</th></tr>"
            
            for feature, stats in analysis['feature_stats'].items():
                null_class = 'warning' if stats['null_percentage'] > 5 else ''
                html += f"""
                <tr>
                    <td>{feature}</td>
                    <td>{stats['mean']:.4f}</td>
                    <td>{stats['std']:.4f}</td>
                    <td>{stats['min']:.4f}</td>
                    <td>{stats['max']:.4f}</td>
                    <td class='{null_class}'>{stats['null_percentage']:.2f}%</td>
                </tr>
                """
            html += "</table>"
        
        html += "</div></body></html>"
        return html

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process cryptocurrency data')
    parser.add_argument('--data_dir', type=str, default='data/raw',
                      help='Directory containing raw data files')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                      help='Directory to save processed data')
    parser.add_argument('--sequence_length', type=int, default=128,
                      help='Length of input sequences')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                      help='Ratio of data to use for training')
    parser.add_argument('--chunk_size', type=int, default=50000,
                      help='Size of data chunks for processing')
    parser.add_argument('--compress', action='store_true',
                      help='Enable data compression')
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = CryptoDataProcessor(
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )
        
        # Process data
        logger.info("Starting data processing...")
        processed_data = processor.process_all_timeframes(
            sequence_length=args.sequence_length,
            train_ratio=args.train_ratio
        )
        
        # Save results
        processor.save_processed_data(
            processed_data,
            compress=args.compress
        )
        
        # Generate report
        processor.generate_report()
        
        logger.info("Processing completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main processing: {e}")
        raise
        
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main()