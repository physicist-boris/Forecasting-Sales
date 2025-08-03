from typing_extensions import Self, override, Tuple
import pandas as pd
import numpy as np
from forecasting_sales.nodes.base.abstract_node import AbstractNode
from forecasting_sales.tools.utils.logging import log_execution


class PreprocessNode(AbstractNode):
    """
    Preprocessing Node
    """
    def __init__(self: Self) -> None:
        pass

    @override
    @log_execution
    def process(
        self: Self, df_to_process: pd.DataFrame,
    ) -> Tuple[pd.DataFrame]:
        """
        Process execution
        """
        df_engineered = self._engineer_features(
            df_to_process
        )
        df_processed = self._create_temporal_features(
            df_engineered
        )
        return (df_processed,)

    @staticmethod
    def _create_temporal_features(df_engineered_feat: pd.DataFrame) -> pd.DataFrame:
        """
        Create meaningful temporal features

        """
        df = df_engineered_feat.copy()
        df['month'] = df['Order Date'].dt.month
        df['year'] = df['Order Date'].dt.year
        # Encodage cyclic du mois (saisonnalité annuelle)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        # Créer les lags pour intégrer l'historique (ici 1 à 12 mois)
        for lag in range(1, 7):
            df[f'sales_lag_{lag}'] = df['Sales'].shift(lag)
        # Tendance (trend) : on peut simplement utiliser l'ordre temporel
        df['trend'] = np.arange(len(df))
        # Supprimer les lignes avec NaN (car lags au début)
        df = df.dropna()
        # Drop non-necessary columns
        df_with_temporal_features = df.drop(columns=['Order Date', 'month', 'year'])
        return df_with_temporal_features

    @staticmethod
    def _engineer_features(
        df_to_prepare:pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Extraction of meaningful features
        """
        # Drop non-necessary columns
        df = df_to_prepare.drop(["Row ID", "Order ID", "Customer ID", "Product ID", "Country",
              "Category", "Ship Mode", "Ship Date", "Customer Name", "Segment", "City",
              "Postal Code", "Region", "Product Name", "Discount", "Profit"], axis=1)
        # Pivot table by Sub-Category and Quantity by month
        df = df.sort_values('Order Date')
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        df_qte_pivot = df.groupby([
            pd.Grouper(key='Order Date', freq='M'),
            'Sub-Category'
        ])['Quantity'].sum().unstack(fill_value=0)
        # Agg time-series Sales by month
        df_ts_month = df.groupby(pd.Grouper(key='Order Date', freq='M'))['Sales'].sum()
        # Merge two aggregations
        df_transformed = df_qte_pivot.join(df_ts_month).reset_index()
        return df_transformed
