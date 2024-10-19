# test_program.py
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from data_transformation import handle_missing_values, encode_categorical_variables, scale_numerical_features, apply_fourier_transformation, apply_pca_transformation
from feature_engineering import calculate_indicators, calculate_wavelet_features, calculate_tsne_features, calculate_kmeans_features
from main import process_chunk, validate_dataframe
from logging_config import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

class TestProgramFunctionality(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        self.sample_data = {
            'Close': [100, 120, 90, 110, 130],
            'High': [110, 130, 100, 120, 140],
            'Low': [90, 100, 80, 90, 110],
            'Volume': [1000, 1200, 900, 1100, 1300],
            'Category': ['A', 'B', 'A', 'B', 'A']
        }
        self.df = pd.DataFrame(self.sample_data)

    def test_handle_missing_values(self):
        # Introduce missing values
        self.df.loc[0, 'Close'] = np.nan
        
        # Test handling missing values
        handled_df = handle_missing_values(self.df)
        self.assertFalse(handled_df.isnull().values.any())

    def test_encode_categorical_variables(self):
        # Test categorical encoding
        encoded_df = encode_categorical_variables(self.df, ['Category'])
        self.assertIn('Category_A', encoded_df.columns)
        self.assertIn('Category_B', encoded_df.columns)

    def test_scale_numerical_features(self):
        # Test numerical scaling
        scaled_df = scale_numerical_features(self.df, ['Close', 'Volume'])
        self.assertTrue(all(scaled_df[['Close', 'Volume']].apply(lambda x: x.between(-3, 3)).all()))

    def test_apply_fourier_transformation(self):
        # Test Fourier Transformation
        fourier_features = apply_fourier_transformation(self.df['Close'])
        self.assertIn('FFT_Real', fourier_features.columns)
        self.assertIn('FFT_Imag', fourier_features.columns)

    def test_apply_pca_transformation(self):
        # Test PCA Transformation
        pca_features = apply_pca_transformation(self.df[['Close', 'Volume']])
        self.assertEqual(pca_features.shape[1], 2)  # Default n_components=2

    def test_calculate_indicators(self):
        # Test technical indicators calculation
        indicators_df = calculate_indicators(self.df)
        self.assertIn('RSI', indicators_df.columns)
        self.assertIn('MACD', indicators_df.columns)

    def test_calculate_wavelet_features(self):
        # Test wavelet features calculation
        wavelet_features = calculate_wavelet_features(self.df['Close'].values)
        self.assertGreater(len(wavelet_features), 0)

    def test_calculate_tsne_features(self):
        # Test t-SNE features calculation
        tsne_features = calculate_tsne_features(self.df[['Close', 'Volume']])
        self.assertEqual(tsne_features.shape[1], 2)  # Default n_components=2

    def test_calculate_kmeans_features(self):
        # Test K-Means clustering
        kmeans_features = calculate_kmeans_features(self.df[['Close', 'Volume']])
        self.assertIn('Cluster', kmeans_features.columns)

    def test_process_chunk(self):
        # Mock file reading for testing process_chunk
        with patch('dask.dataframe.read_csv') as mock_read_csv:
            mock_read_csv.return_value.compute.return_value = self.df
            processed_df = process_chunk(['sample_file.csv'])
            self.assertIsNotNone(processed_df)

    def test_validate_dataframe(self):
        # Test DataFrame validation
        self.assertTrue(validate_dataframe(self.df))

if __name__ == '__main__':
    unittest.main()