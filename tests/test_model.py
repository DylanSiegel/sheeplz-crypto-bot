# File: tests/test_model.py

import unittest
import torch
from models.model import TradingModel

class TestTradingModel(unittest.TestCase):

    def setUp(self):
        self.model = TradingModel(input_size=7, hidden_size=64, num_layers=2, output_size=3, dropout=0.2)

    def test_forward(self):
        # Create a dummy input tensor
        input_tensor = torch.randn(1, 7)
        output = self.model(input_tensor)
        self.assertEqual(output.shape, (1, 3))

    def test_parameters(self):
        # Check if model parameters are correctly initialized
        self.assertEqual(self.model.hidden_size, 64)
        self.assertEqual(self.model.num_layers, 2)
        self.assertEqual(self.model.output_size, 3)

if __name__ == '__main__':
    unittest.main()
