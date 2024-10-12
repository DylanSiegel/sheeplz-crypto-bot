# tests/test_real_time_lnn.py

import unittest
import asyncio
import numpy as np
from unittest.mock import patch, AsyncMock
from src.real_time_lnn import RealTimeLNN
from src.utils import load_config

class TestRealTimeLNN(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Load the configuration for testing
        config = load_config("config/config.yaml")
        self.lnn = RealTimeLNN(config)

    def test_preprocess_data_aggTrade(self):
        data = {
            "e": "aggTrade",
            "p": "50000.00",
            "q": "0.001",
            "f": 100,
            "l": 101,
            "m": True,
            "T": 1616660000000
        }
        processed = self.lnn.preprocess_data(data)
        self.assertEqual(len(processed), self.lnn.model.input_size)

    def test_preprocess_data_invalid(self):
        data = {
            "e": "unknownEvent",
            "invalid_field": "abc",
            "T": 1616660000000
        }
        processed = self.lnn.preprocess_data(data)
        self.assertEqual(len(processed), self.lnn.model.input_size)

    @patch('src.real_time_lnn.RealTimeLNN.predict', return_value=np.array([[0.1]*10]))
    async def test_predict(self, mock_predict):
        # Create a random input sequence matching the required shape
        input_sequence = np.random.rand(self.lnn.sequence_length, self.lnn.model.input_size)
        prediction = await self.lnn.predict(input_sequence)
        mock_predict.assert_called_once_with(input_sequence)
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.shape, (1, self.lnn.model.output_size))

if __name__ == '__main__':
    unittest.main()
