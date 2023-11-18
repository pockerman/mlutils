from mlutils.config import DATA_DIR
from typing import Any
from pathlib import Path
import pandas as pd
import numpy as np


class MLUtilsDataLoader:
    """Simple class that allows to load specific datasets

    """

    DATASETS = ["covid_flu.csv", "compas-scores-two-years.csv",
                "heart_characterization"]

    @staticmethod
    def load(dataset_name: str) -> Any:

        if dataset_name not in MLUtilsDataLoader.DATASETS:
            raise ValueError(f"Dataset {dataset_name} is unknown")

        if dataset_name == "covid_flu.csv":
            return MLUtilsDataLoader.load_covid_flu()

        if dataset_name == "compas-scores-two-years.csv":
            return MLUtilsDataLoader.load_compas_scores_two_years()

        if dataset_name == "heart_characterization":
            return MLUtilsDataLoader.load_heart_characterization()

    @staticmethod
    def load_covid_flu() -> pd.DataFrame:
        data_path = DATA_DIR / "covid_flu.csv"
        return pd.read_csv(data_path)

    @staticmethod
    def load_compas_scores_two_years() -> pd.DataFrame:
        data_path = DATA_DIR / "compas-scores-two-years.csv"
        return pd.read_csv(data_path)

    @staticmethod
    def load_heart_characterization() -> np.ndarray:
        """Data for heart characterization from
        John A. Rice, Mathematical Statistics and Data Analysis, 2nd Edition, Duxbury Press
        chapter 14, section 5
        column 1: Height (in.)
        column 2: Weight (lb)
        column 3: Distance to Pulmonary Artery (cm)
        Returns
        -------

        """
        data = [[42.8, 40.0, 37.0],
                [63.5, 93.5, 49.5],
                [37.5, 35.5, 34.5],
                [39.5, 30.0, 36.0],
                [45.5, 52.0, 43.0],
                [38.5, 17.0, 28.0],
                [43.0, 38.5, 37.0],
                [22.5, 8.5,  20.0],
                [37.0, 33.0, 33.5],
                [23.5, 9.5,  30.5],
                [33.0, 21.0, 38.5],
                [58.0, 79.0, 47.0]]
        return np.ndarray(data)
