import json
import pandas as pd

from typing_extensions import Self, Tuple, override

from forecasting_sales.nodes.base.storage.local_storage import (
    AbstractNode as AbstractStorageNode,
)
from forecasting_sales.tools.data_science.forecasting import ForecastingConfig
from forecasting_sales.tools.utils.filesystem import root_directory


class NodeLocalStorage(AbstractStorageNode):
    """
    Forecasting Local node storage
    """

    CHECKPOINT_FORECASTING = (
        root_directory() / "data" / "03_forecasting" / "forecasting_result.json"
    )

    CHECKPOINT_FORECASTING_FOLDER = root_directory() / "data" / "03_forecasting"

    INPUT_DATA = (
        root_directory()
        / "data"
        / "02_preprocessed_data"
        / "preprocessed_stores_sales_forecasting.csv"
    )

    CONFIG_PATH = (
        root_directory() / "conf" / "local" / "forecasting" / "forecasting_config.json"
    )

    def __init__(self: Self) -> None:
        self.input_data = NodeLocalStorage.INPUT_DATA
        self.checkpoint_forecasting = NodeLocalStorage.CHECKPOINT_FORECASTING
        self.checkpoint_forecasting_folder = (
            NodeLocalStorage.CHECKPOINT_FORECASTING_FOLDER
        )
        self.config_path = NodeLocalStorage.CONFIG_PATH

    @override
    def save_checkpoint(self: Self, dictionary_of_results: dict[str, int]) -> None:
        with open(
            self.checkpoint_forecasting,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(dictionary_of_results, f, ensure_ascii=False)

    @override
    def load_checkpoint(self: Self) -> pd.DataFrame:
        with open(
            self.checkpoint_forecasting,
            "r",
            encoding="utf-8",
        ) as f:
            dictionary_of_results = json.load(f)
        return dictionary_of_results

    def _load_artefact(self: Self) -> ForecastingConfig:
        with open(self.config_path, encoding="utf-8") as file:
            forecasting_config = ForecastingConfig.from_dict(json.load(file))  # type: ignore[attr-defined]
        return forecasting_config  # type: ignore[no-any-return]

    @override
    def load_source(self: Self) -> Tuple[pd.DataFrame, ForecastingConfig]:
        forecasting_params = self._load_artefact()
        df_preprocessed = pd.read_csv(self.input_data, encoding="ISO-8859-1")
        return (df_preprocessed, forecasting_params)
