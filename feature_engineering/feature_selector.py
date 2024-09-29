# File: feature_engineering/feature_selector.py

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

class FeatureSelector:
    def __init__(self, config):
        self.config = config
        self.selector = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        if self.config.method == 'SelectFromModel':
            self.selector = SelectFromModel(
                RandomForestRegressor(n_estimators=100, random_state=42),
                threshold=self.config.threshold,
                max_features=self.config.max_features
            )
        else:
            raise ValueError(f"Unsupported feature selection method: {self.config.method}")

        self.selector.fit(X, y)
        return X.iloc[:, self.selector.get_support()]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.selector is None:
            raise RuntimeError("Feature selector has not been fitted.")
        return X.iloc[:, self.selector.get_support()]