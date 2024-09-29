# File: src/data/feature_store.py

import pandas as pd
import os
from typing import Optional

class FeatureStore:
    """
    Manages the storage and retrieval of features.
    """

    def __init__(self, feature_save_path: str):
        """
        Initializes the FeatureStore.

        Args:
            feature_save_path (str): Path to save the features.
        """
        self.feature_save_path = feature_save_path
        os.makedirs(self.feature_save_path, exist_ok=True)

    def save_features(self, df: pd.DataFrame, filename: str):
        """
        Saves the feature DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): Feature DataFrame.
            filename (str): Name of the file to save.
        """
        filepath = os.path.join(self.feature_save_path, filename)
        df.to_csv(filepath, index=False)

    def load_features(self, filename: str) -> pd.DataFrame:
        """
        Loads features from a CSV file.

        Args:
            filename (str): Name of the file to load.

        Returns:
            pd.DataFrame: Loaded feature DataFrame.
        """
        filepath = os.path.join(self.feature_save_path, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Feature file not found: {filepath}")
        return pd.read_csv(filepath)

    def list_features(self) -> list:
        """
        Lists all saved feature files.

        Returns:
            list: List of feature filenames.
        """
        return os.listdir(self.feature_save_path)