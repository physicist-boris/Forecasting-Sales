"""
Entrypoint for pipeline
"""
from enum import Enum
import logging
from datetime import datetime
from pathlib import Path
import uuid

from dataclasses import dataclass
from typing import Optional, List, Any, Tuple, Iterable

from pyspark.sql import SparkSession

log_path = Path(__file__).parent.parent / "logs" / "journals"
log_name =  f"run_execution_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"


logging.getLogger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Pipeline:
    """
    Pipeline class containing the nodes to execute
    """
    local_storage_nodes: Iterable[Any]
    databricks_storage_nodes: Iterable[Any]
    process_nodes: Iterable[Any]

class StorageTypes(Enum):
    """
    Available storage types
    """
    DATABRICKS = "databricks"
    LOCAL = "local"


class NodesManager:
    """
    Nodes manager to initialized and execute the nodes
    """

    @staticmethod
    def execute_nodes(nodes: List[Tuple[Any, Any]], init_params: Tuple[Any]) -> None:
        """
        Method to executes nodes specified. 
        The init_params is for the storage classes. 
        """
        for storage_node, process_node in nodes:
            storage = storage_node(*init_params)
            sources = storage.load_source()
            results = process_node().process(*sources)
            storage.save_checkpoint(*results)

    @staticmethod
    def define_init_parameters(storage: str = StorageTypes.LOCAL,
                               spark_session: Optional[SparkSession] = None,
                               run_id: Optional[str] = None) -> Tuple[Optional[Any],...]:
        """
        Define initialize parameters for the nodes
        """
        init_params: Tuple[Optional[Any],...]
        if storage == StorageTypes.DATABRICKS:
            if spark_session is None:
                spark_session = SparkSession.builder.appName('pipeline_cli').getOrCreate()
            init_params = (spark_session,run_id)

        elif storage == StorageTypes.LOCAL:
            init_params = ()

        else:
            raise ValueError(f"The value provided {storage} is not managed. The storage " \
                            f"types managed are {[StorageTypes.DATABRICKS.value, StorageTypes.LOCAL.value]}" \
                             " for the storage types" \
                            f"{[element.value for element in StorageTypes]}")                 

        return init_params


    @staticmethod
    def execute_pipeline(pipeline: Pipeline,
                         storage: str = StorageTypes.LOCAL,
                         spark_session: Optional[SparkSession] = None,
                         source_run_id: Optional[str] = None) -> None:
        """
        Execute the pipeline
        """
        
        if storage == StorageTypes.DATABRICKS:
            import mlflow
            with mlflow.start_run() as run:
                storage_nodes = pipeline.databricks_storage_nodes
                logging.getLogger("py4j").setLevel(logging.ERROR)
                logging.getLogger().addHandler(logging.StreamHandler())

            raise ValueError("Databricks is not currently accepted as a storage type.")

        elif storage == StorageTypes.LOCAL:
            source_run_id = str(uuid.uuid4().hex)
            storage_nodes = pipeline.local_storage_nodes
            logging.basicConfig(
                filename=str(log_path / log_name),
                format="%(asctime)s %(message)s", filemode="w"
            )

        else:
            raise ValueError(f"The value provided {storage} is not valid. " \
                            f"The storage types are {[element.value for element in StorageTypes]}")

        run_id = source_run_id if source_run_id is not None else run.info.run_id
        init_params = NodesManager.define_init_parameters(storage, spark_session, run_id)
        if len(storage_nodes) != len(pipeline.process_nodes):
            raise ValueError(f"The number of precess nodes and storage nodes need to be the same. " \
                              f"The process nodes are {[x.__qualname__ for x in pipeline.process_nodes]} and " \
                              f"the storage nodes are {[x.__qualname__ for x in storage_nodes]}")

        nodes = list(zip(storage_nodes, pipeline.process_nodes))
        NodesManager.execute_nodes(nodes, init_params) # type: ignore
