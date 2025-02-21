import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    AdaBoostRegressor,
    RandomForestRegressor,
)

from config import EVAL_FUNC


class MachineLearningModel:
    """The utils for machine learning model training and testing"""

    def __init__(self):
        # You should replace the None with the model you want to use while defining new Machine Learning Model
        self.model = None
        super().__init__()

    def __call__(self, x):
        """Calculate the prediction."""
        if self.model is None:
            raise ValueError("The model is not initiated.")
        x = x.reshape(x.shape[0], -1)
        return self.model.predict(x)

    def fit(self, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray, eval_func):
        """Model training."""
        if self.model is None:
            raise ValueError("The model is not initiated.")
        train_x = train_x.reshape(train_x.shape[0], -1)
        test_x = test_x.reshape(test_x.shape[0], -1)
        self.model.fit(train_x, train_y.ravel())
        return eval_func(self.model.predict(train_x), train_y), eval_func(
            self.model.predict(test_x), test_y
        )

    def eval(self, x: np.ndarray, y: np.ndarray, eval_func):
        """Model evaluation."""
        if self.model is None:
            raise ValueError("The model is not initiated.")
        if eval_func not in EVAL_FUNC:
            raise ValueError("Invalid evaluation function.")
        x = x.reshape(x.shape[0], -1)
        return eval_func(self.model.predict(x), y)

class XGBoost(MachineLearningModel):
    """XGBoost model for the project."""

    def __init__(
        self,
        param_xgb,
    ):
        super().__init__()
        self.model = XGBRegressor(**param_xgb)
        self.type = "XGBoost"
        print("---XGBoost model initiated.---")


class GBDT(MachineLearningModel):
    """GBDT model for the project."""

    def __init__(self, param_gbdt):
        super().__init__()
        self.model = GradientBoostingRegressor(**param_gbdt)
        self.type = "GBDT"
        print("---GBDT model initiated.---")


class AdaBoost(MachineLearningModel):
    """AdaBoost model for the project."""

    def __init__(self, param_ada):
        super().__init__()
        self.model = AdaBoostRegressor(**param_ada)
        self.type = "AdaBoost"
        print("---AdaBoost model initiated.---")


class RandomForest(MachineLearningModel):
    """RandomForest model for the project."""

    def __init__(self, param_rf):
        super().__init__()
        self.model = RandomForestRegressor(**param_rf)
        self.type = "RandomForest"
        print("---RandomForest model initiated.---")


if __name__ == "__main__":
    print("Machine learning models are defined in this file.")
