# models/trainer.py

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import pandas as pd
from src.data.data_loader import TradingDataset
from src.models.base_model import TradingModel
from src.utils.utils import get_logger
from src.features.feature_selector import FeatureSelector

logger = get_logger()

logger = get_logger()

class TradingLitModel(pl.LightningModule):
    """
    PyTorch Lightning module for trading.
    """

    def __init__(self, model: TradingModel, learning_rate: float, loss_fn, optimizer_cls):
        super(TradingLitModel, self).__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.optimizer_cls = optimizer_cls

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        X, y = batch
        preds = self.forward(X)
        loss = self.loss_fn(preds, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(self.model.parameters(), lr=self.learning_rate)
        return optimizer

def train_model(config, train_df: pd.DataFrame, target_df: pd.Series):
    """
    Trains the trading model using PyTorch Lightning.

    Args:
        config (dict): Configuration dictionary.
        train_df (pd.DataFrame): Training feature data.
        target_df (pd.Series): Training target data.
    """
    # Feature Selection
    feature_selector = FeatureSelector(threshold=config['feature_selection']['threshold'],
                                       max_features=config['feature_selection']['max_features'])
    X_selected = feature_selector.fit_transform(train_df, target_df)
    logger.info(f"Selected features: {X_selected.columns.tolist()}")

    # Dataset and DataLoader
    dataset = TradingDataset(X_selected, target_df)
    dataloader = DataLoader(dataset, batch_size=config['model']['batch_size'], shuffle=True)

    # Model
    model = TradingModel(input_size=X_selected.shape[1],
                         hidden_size=config['model']['hidden_size'],
                         output_size=config['model']['output_size'])
    
    # Lightning Module
    lit_model = TradingLitModel(model=model,
                                learning_rate=config['model']['learning_rate'],
                                loss_fn=torch.nn.MSELoss(),
                                optimizer_cls=torch.optim.Adam)

    # Trainer
    trainer = pl.Trainer(max_epochs=config['model']['epochs'],
                         gpus=1 if torch.cuda.is_available() else 0,
                         logger=True)

    # Train
    trainer.fit(lit_model, dataloader)

    # Save the trained model
    torch.save(model.state_dict(), config['model']['model_save_path'] + "trading_model.pth")
    logger.info("Model training completed and saved.")

# Example usage
# train_model(config, train_features, train_targets)
