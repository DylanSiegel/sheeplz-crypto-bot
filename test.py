# File: test.py

import unittest
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from unittest.mock import patch, MagicMock
import joblib
import os
import shutil
import multiprocessing  # Import multiprocessing
from multiprocessing import Pool, cpu_count  # Import Pool and cpu_count

# Import functions and classes from lnn_module.py
from lnn_module import (
    load_and_preprocess,
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    create_sequences,
    FinancialDataset,
    LNNModel,
    SEQ_LENGTH,
    BATCH_SIZE,
    SCALER_DIR,
    MODEL_DIR
)

# Define constants for testing
TEST_PROCESSED_DATA_DIR = 'test_data_final'
TEST_RAW_DATA_DIR = 'test_data_raw'
TEST_SCALER_DIR = 'test_scalers'
TEST_MODEL_DIR = 'test_models'

# Define NUM_WORKERS
NUM_WORKERS = cpu_count()

class TestLNNProgram(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up synthetic datasets for testing purposes.
        This method runs once before all tests.
        """
        os.makedirs(TEST_PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(TEST_RAW_DATA_DIR, exist_ok=True)
        os.makedirs(TEST_SCALER_DIR, exist_ok=True)
        os.makedirs(TEST_MODEL_DIR, exist_ok=True)

        # Create synthetic raw data for each timeframe
        cls.synthetic_raw_data = {}
        timeframes = ['15m', '1h', '4h', '1d']
        for timeframe in timeframes:
            raw_filename = f"btc_{timeframe}_data_2018_to_2024-2024-10-10.csv"
            raw_path = os.path.join(TEST_RAW_DATA_DIR, raw_filename)
            freq = '15T' if timeframe == '15m' else 'H' if timeframe == '1h' else '4H' if timeframe == '4h' else 'D'
            data = {
                'Open time': pd.date_range(start='2022-01-01', periods=1000, freq=freq),
                'Open': np.random.rand(1000).astype(np.float32) * 10000,
                'High': np.random.rand(1000).astype(np.float32) * 10000,
                'Low': np.random.rand(1000).astype(np.float32) * 10000,
                'Close': np.random.rand(1000).astype(np.float32) * 10000,
                'Volume': np.random.rand(1000).astype(np.float32) * 100,
                'Close time': pd.date_range(start='2022-01-01', periods=1000, freq=freq),
                'Quote asset volume': np.random.rand(1000).astype(np.float32) * 1e6,
                'Number of trades': np.random.randint(100, 1000, size=1000),
                'Taker buy base asset volume': np.random.rand(1000).astype(np.float32) * 100,
                'Taker buy quote asset volume': np.random.rand(1000).astype(np.float32) * 1e5,
                'Ignore': [0]*1000
            }
            df_raw = pd.DataFrame(data)
            df_raw.to_csv(raw_path, index=False)
            cls.synthetic_raw_data[timeframe] = raw_path

        # Preprocess synthetic raw data
        from preprocess import preprocess_file  # Import the preprocess_file function

        for timeframe, raw_path in cls.synthetic_raw_data.items():
            processed_filename = f"processed_data_{timeframe}.csv.gz"
            processed_path = os.path.join(TEST_PROCESSED_DATA_DIR, processed_filename)
            preprocess_file(timeframe, raw_path, processed_path)

    @classmethod
    def tearDownClass(cls):
        """
        Clean up the synthetic datasets after all tests have run.
        """
        shutil.rmtree(TEST_PROCESSED_DATA_DIR, ignore_errors=True)
        shutil.rmtree(TEST_RAW_DATA_DIR, ignore_errors=True)
        shutil.rmtree(TEST_SCALER_DIR, ignore_errors=True)
        shutil.rmtree(TEST_MODEL_DIR, ignore_errors=True)

    def test_load_and_preprocess(self):
        """
        Test the load_and_preprocess function to ensure it correctly loads data,
        handles missing values, and scales features for all timeframes.
        """
        timeframes = ['15m', '1h', '4h', '1d']
        for timeframe in timeframes:
            with self.subTest(timeframe=timeframe):
                processed_path = os.path.join(TEST_PROCESSED_DATA_DIR, f"processed_data_{timeframe}.csv.gz")
                feature_cols = FEATURE_COLUMNS[timeframe]
                target_cols = TARGET_COLUMNS[timeframe]
                scaler_path = os.path.join(TEST_SCALER_DIR, f"scaler_{timeframe}.save")

                df_scaled = load_and_preprocess(processed_path, feature_cols, target_cols, scaler_path)

                # Check for missing values
                self.assertFalse(df_scaled.isnull().values.any(), f"Missing values in {timeframe} data.")

                # Check if scaler was saved
                self.assertTrue(os.path.exists(scaler_path), f"Scaler not saved for {timeframe} data.")

                # Load scaler and verify scaling
                scaler = joblib.load(scaler_path)
                original_features = pd.read_csv(processed_path, compression='gzip')[feature_cols].astype(np.float32)
                scaled_features = scaler.transform(original_features)
                np.testing.assert_array_almost_equal(
                    df_scaled[feature_cols].values,
                    scaled_features,
                    decimal=5,
                    err_msg=f"Feature scaling incorrect for {timeframe} data."
                )

    def test_create_sequences(self):
        """
        Test the create_sequences function to ensure it correctly creates input sequences and targets for all timeframes.
        """
        timeframes = ['15m', '1h', '4h', '1d']
        for timeframe in timeframes:
            with self.subTest(timeframe=timeframe):
                processed_path = os.path.join(TEST_PROCESSED_DATA_DIR, f"processed_data_{timeframe}.csv.gz")
                feature_cols = FEATURE_COLUMNS[timeframe]
                target_cols = TARGET_COLUMNS[timeframe]
                scaler_path = os.path.join(TEST_SCALER_DIR, f"scaler_{timeframe}.save")

                df_scaled = load_and_preprocess(processed_path, feature_cols, target_cols, scaler_path)
                sequences, targets = create_sequences(df_scaled, SEQ_LENGTH, feature_cols, target_cols)

                # Calculate expected number of sequences
                expected_num_sequences = len(df_scaled) - SEQ_LENGTH
                actual_num_sequences = sequences.shape[0]

                self.assertEqual(
                    actual_num_sequences,
                    expected_num_sequences,
                    f"Sequences count mismatch for {timeframe} data. Expected {expected_num_sequences}, got {actual_num_sequences}."
                )

                # Check shapes
                self.assertEqual(
                    sequences.shape[1],
                    SEQ_LENGTH,
                    f"Sequence length incorrect for {timeframe} data."
                )
                self.assertEqual(
                    sequences.shape[2],
                    len(feature_cols),
                    f"Number of features incorrect for {timeframe} data."
                )

                for target in target_cols:
                    self.assertEqual(
                        len(targets[target]),
                        expected_num_sequences,
                        f"Target length mismatch for {timeframe} data."
                    )

    def test_financial_dataset(self):
        """
        Test the FinancialDataset class to ensure it correctly retrieves sequences and targets for all timeframes.
        """
        timeframes = ['15m', '1h', '4h', '1d']
        for timeframe in timeframes:
            with self.subTest(timeframe=timeframe):
                processed_path = os.path.join(TEST_PROCESSED_DATA_DIR, f"processed_data_{timeframe}.csv.gz")
                feature_cols = FEATURE_COLUMNS[timeframe]
                target_cols = TARGET_COLUMNS[timeframe]
                scaler_path = os.path.join(TEST_SCALER_DIR, f"scaler_{timeframe}.save")

                df_scaled = load_and_preprocess(processed_path, feature_cols, target_cols, scaler_path)
                sequences, targets = create_sequences(df_scaled, SEQ_LENGTH, feature_cols, target_cols)
                dataset = FinancialDataset(sequences, {target_cols[0]: targets[target_cols[0]]})  # Testing first target

                # Test dataset length
                self.assertEqual(
                    len(dataset),
                    len(sequences),
                    f"Dataset length mismatch for {timeframe} data."
                )

                # Test getting an item
                sample_seq, sample_target = dataset[0]
                np.testing.assert_array_almost_equal(
                    sample_seq.numpy(),
                    sequences[0],
                    decimal=5,
                    err_msg=f"Sequence data incorrect for {timeframe} data."
                )
                self.assertAlmostEqual(
                    sample_target.item(),
                    targets[target_cols[0]][0],
                    places=6,
                    msg=f"Target value incorrect for {timeframe} data."
                )

    def test_lnn_model_forward(self):
        """
        Test the LNNModel's forward pass to ensure it produces outputs of the correct shape for all timeframes.
        """
        timeframes = ['15m', '1h', '4h', '1d']
        for timeframe in timeframes:
            with self.subTest(timeframe=timeframe):
                input_size = len(FEATURE_COLUMNS[timeframe])
                hidden_size = 128
                num_layers = 2
                output_size = 1

                model = LNNModel(input_size, hidden_size, num_layers, output_size)
                model.eval()  # Set model to evaluation mode

                # Create a dummy input tensor with batch_size=2
                dummy_input = torch.randn(2, SEQ_LENGTH, input_size)
                with torch.no_grad():
                    output = model(dummy_input)

                # Check output shape
                self.assertEqual(
                    output.shape,
                    (2, output_size),
                    f"Model output shape incorrect for {timeframe} data."
                )

    def test_full_pipeline(self):
        """
        Test the full data loading, preprocessing, dataset creation, and model initialization pipeline for all timeframes.
        This ensures that all components work together without errors.
        """
        timeframes = ['15m', '1h', '4h', '1d']
        for timeframe in timeframes:
            with self.subTest(timeframe=timeframe):
                try:
                    processed_path = os.path.join(TEST_PROCESSED_DATA_DIR, f"processed_data_{timeframe}.csv.gz")
                    feature_cols = FEATURE_COLUMNS[timeframe]
                    target_cols = TARGET_COLUMNS[timeframe]
                    scaler_path = os.path.join(TEST_SCALER_DIR, f"scaler_{timeframe}.save")
                    model_save_path = os.path.join(TEST_MODEL_DIR, f"lnn_model_{timeframe}.pth")

                    # Load and preprocess
                    df_scaled = load_and_preprocess(processed_path, feature_cols, target_cols, scaler_path)

                    # Create sequences and targets
                    sequences, targets = create_sequences(df_scaled, SEQ_LENGTH, feature_cols, target_cols)

                    # Create Dataset and DataLoader
                    dataset = FinancialDataset(sequences, {target_cols[0]: targets[target_cols[0]]})  # Using first target
                    data_loader = DataLoader(
                        dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=NUM_WORKERS,
                        pin_memory=True
                    )

                    # Initialize model
                    input_size = len(feature_cols)
                    hidden_size = 128
                    num_layers = 2
                    output_size = 1

                    model = LNNModel(input_size, hidden_size, num_layers, output_size)
                    criterion = torch.nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                    # Perform a single training step
                    model.train()
                    for sequences_batch, targets_batch in data_loader:
                        # Forward pass
                        outputs = model(sequences_batch)
                        loss = criterion(outputs, targets_batch)

                        # Backward pass and optimization
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        break  # Only one batch for testing

                    # Check if loss is a finite number
                    self.assertTrue(np.isfinite(loss.item()), f"Loss is not finite for {timeframe} data.")

                except Exception as e:
                    self.fail(f"Full pipeline test failed for {timeframe} with exception: {e}")

    @patch('joblib.dump')
    def test_scaler_save(self, mock_joblib_dump):
        """
        Test that the scaler is saved correctly during preprocessing for all timeframes.
        """
        timeframes = ['15m', '1h', '4h', '1d']
        for timeframe in timeframes:
            with self.subTest(timeframe=timeframe):
                processed_path = os.path.join(TEST_PROCESSED_DATA_DIR, f"processed_data_{timeframe}.csv.gz")
                feature_cols = FEATURE_COLUMNS[timeframe]
                target_cols = TARGET_COLUMNS[timeframe]
                scaler_path = os.path.join(TEST_SCALER_DIR, f"scaler_{timeframe}.save")

                load_and_preprocess(processed_path, feature_cols, target_cols, scaler_path)

                # Check that joblib.dump was called to save the scaler
                mock_joblib_dump.assert_called_with(
                    unittest.mock.ANY,  # The scaler object
                    scaler_path
                )

    def test_model_save(self):
        """
        Test that the model can be saved without errors for all timeframes.
        """
        timeframes = ['15m', '1h', '4h', '1d']
        for timeframe in timeframes:
            with self.subTest(timeframe=timeframe):
                model = LNNModel(
                    input_size=len(FEATURE_COLUMNS[timeframe]),
                    hidden_size=128,
                    num_layers=2,
                    output_size=1
                )
                model_save_path = os.path.join(TEST_MODEL_DIR, f"lnn_model_{timeframe}.pth")
                try:
                    torch.save(model.state_dict(), model_save_path)
                    self.assertTrue(os.path.exists(model_save_path), f"Model not saved correctly for {timeframe} data.")
                except Exception as e:
                    self.fail(f"Model save test failed for {timeframe} with exception: {e}")

if __name__ == '__main__':
    unittest.main()
