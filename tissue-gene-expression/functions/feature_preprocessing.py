from typing import Union

import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd

from UpperQuartile import UpperQuartile


def feature_preprocessing(feature_matrix: Union[dd.DataFrame, pd.DataFrame],
                          is_dask: bool = False) -> Union[dd.DataFrame, pd.DataFrame]:
    """Apply the following preprocessing steps on a Dask Dataframe representing a gene feature matrix: upper quartile
    normalization; select the 25% of genes with highest mean gene expression and variance; and log2+1 transform
    the remaining expression levels in the matrix.

    Args:
        feature_matrix (Dataframe): a gene expression matrix.
        is_dask (boolean): Whether the Dataframe or Series is a lazy Dask Dataframe (True) or pandas Dataframe (False).

    Returns:
        dd.Dataframe: A Dask Dataframe comprising the processed and selected gene expression levels.
    """
    uq = UpperQuartile(is_dask=is_dask)
    uq_feature_matrix = uq.fit_transform(feature_matrix)
    if is_dask:
        uq_feature_matrix = uq_feature_matrix.persist()

    # For gene features to be maximally informative, we want them to have a minimum expression signal and to
    # vary at least a little across samples.
    mean = uq_feature_matrix.mean(axis=0)
    var = uq_feature_matrix.var(axis=0)
    if is_dask:
        mean, var = mean.persist(), var.persist()

    threshold_feature_matrix = uq_feature_matrix[uq_feature_matrix.columns[(mean > mean.quantile(0.25)) &
                                                                           (var > var.quantile(0.25))]]
    if is_dask:
        threshold_feature_matrix = threshold_feature_matrix.repartition(partition_size='64MB')

    if is_dask:
        log_scaled_feature_matrix = threshold_feature_matrix.applymap(lambda gene: da.log2(gene + 1))
    else:
        log_scaled_feature_matrix = threshold_feature_matrix.applymap(lambda gene: np.log2(gene + 1))

    return log_scaled_feature_matrix
