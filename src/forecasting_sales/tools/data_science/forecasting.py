from dataclasses import dataclass

from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ForecastingConfig:
    """
    Configuration for the infence llm
    """

    n_estimators: int
    learning_rate: int
    max_depth: int
    random_state: int

