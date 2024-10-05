# File: tests/test_train_lnn.py

import pytest
from unittest.mock import patch, MagicMock
from models.lnn.train_lnn import train_lnn

@pytest.mark.asyncio
async def test_train_lnn_success():
    # Mock Config
    with patch("models.utils.config.Config.__init__", return_value=None) as mock_config_init:
        config = Config("configs/config.yaml")
        config.lnn_hidden_size = 64
        config.lnn_training_epochs = 1
        config.training_history_length = 500
        config.lnn_learning_rate = 0.001
        config.lnn_model_path = "models/lnn/lnn_model.pth"

        # Mock data loading
        with patch("pandas.read_csv") as mock_read_csv:
            mock_df = MagicMock()
            mock_df.iloc.__getitem__.return_value = MagicMock()
            mock_read_csv.return_value = mock_df

            # Mock scaler
            with patch("sklearn.preprocessing.MinMaxScaler.fit_transform", return_value=[[1,2,3], [4,5,6]]):
                # Mock torch functionalities
                with patch("torch.tensor") as mock_torch_tensor:
                    mock_torch_tensor.return_value = MagicMock()
                    with patch("torch.optim.Adam") as mock_optimizer:
                        mock_optimizer.return_value.step = MagicMock()
                        mock_optimizer.return_value.zero_grad = MagicMock()
                        mock_optimizer.return_value.zero_grad.return_value = None
                        mock_optimizer.return_value.step.return_value = None

                        # Mock model save
                        with patch("torch.save") as mock_torch_save:
                            await train_lnn()
                            mock_torch_save.assert_called()

def test_train_lnn_file_not_found():
    # Simulate FileNotFoundError during data loading
    with patch("models.utils.config.Config.__init__", return_value=None):
        config = Config("configs/config.yaml")
        config.lnn_training_epochs = 1
        config.lnn_model_path = "models/lnn/lnn_model.pth"

        with patch("pandas.read_csv", side_effect=FileNotFoundError):
            with patch("logging.error") as mock_logging_error:
                with patch("models.lnn.train_lnn.Config", return_value=config):
                    asyncio.run(train_lnn())
                    mock_logging_error.assert_called_with("Training data file not found. Please provide a valid CSV file.")

def test_train_lnn_exception():
    # Simulate a generic exception during training
    with patch("models.utils.config.Config.__init__", return_value=None):
        config = Config("configs/config.yaml")
        config.lnn_training_epochs = 1
        config.lnn_model_path = "models/lnn/lnn_model.pth"

        with patch("pandas.read_csv", return_value=MagicMock()):
            with patch("sklearn.preprocessing.MinMaxScaler.fit_transform", side_effect=Exception("Scaling error")):
                with patch("logging.error") as mock_logging_error:
                    asyncio.run(train_lnn())
                    mock_logging_error.assert_called_with("Error during LNN training: Scaling error")
