import gc

from ..functions.load_data import load_data
from ..functions.feature_preprocessing import feature_preprocessing
from ..functions.pre_ml_processing import pre_ml_processing
from ..functions.xgboost_pipeline import xgboost_pipeline
from ..functions.hpo_pipeline import hpo_pipeline


def spe_pipeline(features_filepath: str, labels_filepath: str, task: str = 'classification',
                 param_dist: dict = None) -> tuple[float, float, float]:

    feature_matrix = load_data(features_filepath, is_dask=False)
    label_vector = load_data(labels_filepath, is_dask=False)

    preprocessed_feature_matrix = feature_preprocessing(feature_matrix, is_dask=False)
    del feature_matrix
    gc.collect()

    X_train, X_test, y_train, y_test = pre_ml_processing(preprocessed_feature_matrix, label_vector, task=task,
                                                         is_dask=False)
    del preprocessed_feature_matrix, label_vector
    gc.collect()

    if param_dist:
        mean_cv_score, std_cv_score, eval_score = hpo_pipeline(param_dist, X_train, X_test, y_train, y_test,
                                                               is_dask=False)
    else:
        mean_cv_score, std_cv_score, eval_score = xgboost_pipeline(X_train, X_test, y_train, y_test, task=task,
                                                                   is_dask=False)

    return mean_cv_score, std_cv_score, eval_score
