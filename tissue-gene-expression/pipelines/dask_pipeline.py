import gc

import dask.array as da
import dask.dataframe as dd

from ..functions.load_data import load_data
from ..functions.feature_preprocessing import feature_preprocessing
from ..functions.hpo_pipeline import hpo_pipeline
from ..functions.pre_ml_processing import pre_ml_processing
from ..functions.xgboost_pipeline import xgboost_pipeline


def df_to_array(preprocessed_feature_matrix: dd.DataFrame, label_vector: dd.Series) -> tuple[da.Array, da.Array]:
    """Convert the preprocessed feature matrix and label vector into lazy Dask Arrays for later compatibility with
    the machine learning pipeline.
    In the specific case of Dask, chunk sizes must align for the two datasets to ensure proper train/test splitting.
    In other words, both arrays must have the same chunk size, and these chunks must have the same start and end
    indices and columns.

    Args:
        preprocessed_feature_matrix (Dask Dataframe): a gene expression matrix that has underwent preprocessing.
        label_vector (Dask Series): A lazy Dask Series containing the label vectors corresponding to the samples
        in the preprocessed feature matrix.

    Returns:
        A tuple of Dask Arrays: The Dask Arrays for the preprocessed feature matrix and label vector, respectively, with
        matching chunks.
    """
    preprocessed_feature_array = preprocessed_feature_matrix.to_dask_array(lengths=True).rechunk('auto')
    label_array = label_vector.to_dask_array(lengths=True).rechunk((preprocessed_feature_array.chunks[0], None))

    return preprocessed_feature_array, label_array


def dask_pipeline(features_filepath: str, labels_filepath: str, task: str='classification',
                  param_dist: dict = None) -> tuple[float, float, float]:
    print("dask pipeline")

    feature_matrix = load_data(features_filepath, is_dask=True).persist()
    label_vector = load_data(labels_filepath, is_dask=True)

    preprocessed_feature_matrix = feature_preprocessing(feature_matrix, is_dask=True)
    del feature_matrix
    gc.collect()

    preprocessed_feature_array, label_array = df_to_array(preprocessed_feature_matrix, label_vector)
    del preprocessed_feature_matrix, label_vector
    gc.collect()

    X_train, X_test, y_train, y_test = pre_ml_processing(preprocessed_feature_array, label_array, is_dask=True,
                                                         task=task)
    del preprocessed_feature_array, label_array
    gc.collect()
    # Persists are very important from here on out for the lazy Dask Arrays because we will use the X and y
    # train/test set arrays a lot.
    X_train, X_test = X_train.persist(), X_test.persist()
    y_train, y_test = y_train.persist(), y_test.persist()

    if param_dist:
        mean_cv_score, std_cv_score, eval_score = hpo_pipeline(param_dist, X_train, X_test, y_train, y_test,
                                                                   is_dask=True)
    else:
        mean_cv_score, std_cv_score, eval_score = xgboost_pipeline(X_train, X_test, y_train, y_test, is_dask=True,
                                                                   task=task)

    return mean_cv_score, std_cv_score, eval_score
