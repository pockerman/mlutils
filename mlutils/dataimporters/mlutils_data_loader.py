from mlutils.config import DATA_DIR
from typing import Any
from pathlib import Path
import pandas as pd

class MLUtilsDataLoader:
    """Simple class that allows to load specific datasets

    """

    DATASETS = ["covid_flu.csv"]

    @staticmethod
    def load(dataset_name: str) -> Any:

        if dataset_name not in MLUtilsDataLoader.DATASETS:
            raise ValueError(f"Dataset {dataset_name} is unknown")

        if dataset_name == "covid_flu.csv":
            return MLUtilsDataLoader.load_covid_flu(Path(DATA_DIR) / dataset_name)


    @staticmethod
    def load_covid_flu(datapath: Path = Path(DATA_DIR) / "covid_flu.csv") -> pd.DataFrame:
        return pd.read_csv(datapath)