import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple
import logging
from enities import SplittingParams


APPLICATION_NAME = 'homework01'
logger = logging.getLogger(APPLICATION_NAME)


def read_data(path: str, **kwargs) -> pd.DataFrame:
    logger.info("Start reading Dataframe from disk...")
    data = pd.read_csv(path, **kwargs)
    logger.info(f"Dataframe have succesfully read, shape: {data.shape}")
    return data


def split_train_test_data(data: pd.DataFrame, params: SplittingParams) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splitting data into train and test
    """
    train_data, test_data = train_test_split(
        data, test_size = params.validation_size, 
        random_state = params.random_state
    )
    return train_data, test_data
