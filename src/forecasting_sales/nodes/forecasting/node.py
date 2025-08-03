import numpy as np
from typing_extensions import Self, override, Tuple, Any
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from forecasting_sales.nodes.base.abstract_node import AbstractNode
from forecasting_sales.tools.data_science.forecasting import ForecastingConfig
from forecasting_sales.tools.utils.logging import log_execution


class ForecastingNode(AbstractNode):
    """
    Forecasting node
    """

    def __init__(self: Self) -> None:
        pass

    @override
    @log_execution
    def process(
        self: Self, df_to_train: pd.DataFrame, config_params: ForecastingConfig
    ) -> Tuple[dict[str, Any]]:
        """
        Process execution
        """
        X = df_to_train.drop(columns=["Sales"])
        y = df_to_train["Sales"]
        tscv = TimeSeriesSplit(n_splits=6)
        # Training and predicting
        dict_of_results = {"MAE": [], "MAPE(en %)": [], "rmse": []}  # type: ignore[var-annotated]
        for fold, (train_index, test_index) in enumerate(tscv.split(X), 1):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            training_data_set = (X_train, X_test, y_train, y_test)
            xgb_model = self._train(training_data_set, config_params)
            self._predict(xgb_model, X_test, y_test, dict_of_results)
        # keeping the last fold (production like)
        dict_of_results["MAE"] = dict_of_results["MAE"][-1]
        dict_of_results["MAPE(en %)"] = dict_of_results["MAPE(en %)"][-1]
        dict_of_results["rmse"] = dict_of_results["rmse"][-1]
        return (dict_of_results,)

    @staticmethod
    def _train(
        training_set: Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
        training_params: ForecastingConfig,
    ) -> xgb.XGBRegressor:
        """
        Method for training
        """
        model = xgb.XGBRegressor(
            objective="reg:absoluteerror",
            n_estimators=training_params.n_estimators,
            learning_rate=training_params.learning_rate,
            max_depth=training_params.max_depth,
            random_state=training_params.random_state,
            # early_stopping_rounds=30
        )
        model.fit(
            training_set[0],
            training_set[2],
            eval_set=[(training_set[1], training_set[3])],
            verbose=False,
        )
        return model

    @staticmethod
    def _predict(
        xgb_regressor: xgb.XGBRegressor,
        X_test_data: pd.DataFrame,
        y_test_data: pd.DataFrame,
        dictionary_of_results: dict[str, list[int]],
    ) -> None:
        """
        Method for predicting and calculating prediction scores
        """
        preds = xgb_regressor.predict(X_test_data)
        mae = mean_absolute_error(y_test_data, preds)
        mask = y_test_data != 0
        mape = (
            np.mean(np.abs((y_test_data[mask] - preds[mask]) / y_test_data[mask])) * 100
        )
        rmse = np.sqrt(mean_squared_error(y_test_data, preds))
        dictionary_of_results["MAE"].append(mae)
        dictionary_of_results["MAPE(en %)"].append(mape)
        dictionary_of_results["rmse"].append(rmse)
