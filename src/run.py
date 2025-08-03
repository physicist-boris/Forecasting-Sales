import fire
from forecasting_sales.pipeline_manager import Pipeline, NodesManager, StorageTypes
from forecasting_sales.nodes.preprocessing.storage.local_storage import NodeLocalStorage as PreprocessingLocalStorage
from forecasting_sales.nodes.preprocessing.node import PreprocessNode
from forecasting_sales.nodes.forecasting.storage.local_storage import NodeLocalStorage as ForecastingLocalStorage
from forecasting_sales.nodes.forecasting.node import ForecastingNode



class Pipelines:
    """
    Class defining the cli interface
    """

    #  pylint: disable=too-few-public-methods
    @staticmethod
    def all(storage: StorageTypes = StorageTypes.LOCAL) -> None:
        """
        cli definition to execute node
        """
        pipeline = Pipeline(
            local_storage_nodes=[PreprocessingLocalStorage, ForecastingLocalStorage],
            databricks_storage_nodes=[],
            process_nodes=[PreprocessNode, ForecastingNode],
        )
        NodesManager.execute_pipeline(pipeline, storage)

    @staticmethod
    def test_training() -> None:
        print("hello world")


if __name__ == "__main__":
    fire.Fire(Pipelines)
