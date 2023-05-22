import logging

from pandas import DataFrame

from exceptions.custom_exceptions import NullDfException

LOGGER = logging.getLogger(__name__)


def normalize_column(df: DataFrame, column: str) -> DataFrame:
    """
    Normalizes columns in a DataFrame

    Args:
        DataFrame: The input DataFrame
        str:  Name of the column to normalize

    Returns:
        DataFrame: A new DataFrame with the specified column normalized between 0 and 1

    Raises:
        NullDfException: If the input DataFrame is None
        ValueError: If the specified column does not exist in the DataFrame

    """
    if df is None:
        LOGGER.warning("DataFrame is None")
        raise NullDfException("DF is None")
    elif column not in df.columns:
        LOGGER.warning(
            "DataFrame empty or column does not exist in the given DataFrame"
        )
    else:
        df[column] = (df[column] - df[column].min()) / (
            df[column].max() - df[column].min()
        )
    return df
