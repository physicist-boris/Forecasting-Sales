import pandas as pd

from typing_extensions import Self, override, Tuple

from forecasting_sales.nodes.base.storage.local_storage import (
    AbstractNode as AbstractStorageNode,
)
from forecasting_sales.tools.utils.filesystem import root_directory


class NodeLocalStorage(AbstractStorageNode):
    """
    Preprocess Local node storage
    """

    CHECKPOINT_PREPARED_DATA = (
        root_directory() / "data" / "02_preprocessed_data" / "preprocessed_stores_sales_forecasting.csv"
    )

    INPUT_DATA = root_directory() / "data" / "01_extracted_data" / "stores_sales_forecasting.csv"


    def __init__(self: Self) -> None:
        self.input_data = NodeLocalStorage.INPUT_DATA
        self.checkpoint_prepared_data = NodeLocalStorage.CHECKPOINT_PREPARED_DATA


    @override
    def save_checkpoint(self: Self, df_to_save: pd.DataFrame) -> None:
        df_to_save.to_csv(self.checkpoint_prepared_data, index=False, encoding="ISO-8859-1")
        

    @override
    def load_checkpoint(self: Self) -> pd.DataFrame:
        df_checked = pd.read_csv(self.checkpoint_prepared_data, encoding="ISO-8859-1")
        return df_checked


    @override
    def load_source(self: Self) -> Tuple[pd.DataFrame]:
        df = pd.read_csv(self.input_data, encoding="ISO-8859-1")
        return (df,)
