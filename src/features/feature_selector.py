# File: src/features/feature_selection.py

from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from typing import Optional
import pandas as pd

class FeatureSelector:
    def __init__(self, method: str = "SelectFromModel", threshold: Optional[float] = None, max_features: Optional[int] = None):
        self.method = method
        self.threshold = threshold
        self.max_features = max_features
        self.selector = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        if self.method == "SelectFromModel":
            estimator = Lasso(alpha=0.01)
            self.selector = SelectFromModel(estimator, threshold=self.threshold)
        elif self.method == "RFE":
            estimator = RandomForestClassifier(n_estimators=100)
            self.selector = RFE(estimator, n_features_to_select=self.max_features)
        else:
            raise ValueError(f"Unsupported feature selection method: {self.method}")

        self.selector.fit(X, y)
        selected_features = X.columns[self.selector.get_support()]
        return X[selected_features]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.selector:
            raise RuntimeError("FeatureSelector not fitted yet.")
        selected_features = X.columns[self.selector.get_support()]
        return X[selected_features]

    def get_selected_features(self) -> list:
        if not self.selector:
            raise RuntimeError("FeatureSelector not fitted yet.")
        return list(self.selector.get_support(indices=True))