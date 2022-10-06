from typing import Union

import pandas as pd
import dask.dataframe as dd


def load_data(filepath: str, is_dask: bool = False,
              compute: bool = False) -> Union[dd.DataFrame, dd.Series, pd.DataFrame, pd.Series]:
    """Load a parquet or csv file into a Dataframe or Series that is either SPE- (pandas, scikit-learn, etc.) or
    Dask-compatible. Dask-compatible data is loaded lazily as a task graph rather than eagerly loaded into memory.

    Args:
        filepath (string): The filepath with the file to load into a Dataframe or Series.
        is_dask (boolean): Whether the Dataframe or Series is to be loaded for use with Dask (True) or SPE (False).
        compute (boolean): Whether to eagerly compute the Dataframe or Series or build a lazy task graph. Only available
        to Dask task graphs.

    Returns:
        (dd.Dataframe or dd.Series) or (pd.Dataframe or pd.Series): A Dataframe or Series containing the data
         stored in the filepath, either for use with Dask (Dask Dataframe) or SPE (Pandas Dataframe).
    """
    if is_dask:
        if 'parquet' in filepath.split('.')[-1]:
            df = dd.read_parquet(filepath)
        else:
            # assume_missing=True ensures that all integer columns are interpreted as floats
            df = dd.read_csv(filepath, assume_missing=True, sample=2000000)
        if compute:
            df = df.compute()
    else:
        df = pd.read_csv(filepath)

    return df