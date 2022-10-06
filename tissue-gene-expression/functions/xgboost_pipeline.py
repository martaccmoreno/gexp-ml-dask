from typing import Union

import dask.array as da
import numpy as np
import xgboost as xgb

from sklearn.model_selection import cross_val_score
from dask_ml.metrics import accuracy_score as dask_accuracy_score
from dask_ml.metrics import r2_score as dask_r2_score
from sklearn.metrics import accuracy_score, r2_score

from dask_cv import dask_cv


def xgboost_pipeline(X_train: Union[da.Array, np.array], X_test: Union[da.Array, np.array],
                     y_train: Union[da.Array, np.array], y_test: Union[da.Array, np.array],
                     task: str='classification', is_dask: bool = False) -> tuple[float, float, da.Array]:
    """Pipeline for passing prepared feature and label train/test sets through a XGBoost estimator.
    Works for both classification and regression.

    Args:
        X_train, X_test: train and test gene expression feature arrays.
        y_train, y_test: train and test label arrays.
        task (string): The machine learning task to perform: classification or regression.
        is_dask (boolean): Whether the Arrays are a lazy Dask Arrays (True) or numpy Arrays (False).

    Returns:
        Mean Cross-Validation (CV) Score (float)
        Standard Deviation for the CV Score (float)
        Evaluation Score for the model on the test set (float)
    """
    if task.lower() == 'classification' or task.lower() == 'class':
        bst_cv = xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False, n_jobs=-1)
        if is_dask:
            bst_eval = xgb.dask.DaskXGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1)
            regression = False
        else:
            bst_eval = xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False, n_jobs=-1)
            scoring = 'accuracy'
    elif task.lower() == 'regression' or task.lower() == 'regress':
        bst_cv = xgb.XGBRegressor(random_state=42, eval_metric='rmse', n_jobs=-1)
        if is_dask:
            bst_eval = xgb.XGBRegressor(random_state=42, eval_metric='rmse', n_jobs=-1)
            regression = True
        else:
            bst_eval = xgb.dask.DaskXGBRegressor(random_state=42, eval_metric='rmse', n_jobs=-1)
            scoring = 'r2'
    else:
        raise ValueError('Invalid ML task!')

    if is_dask:
        cv_scores = dask_cv(bst_cv, X_train, y_train, regression=regression)
    else:
        cv_scores = cross_val_score(bst_cv, X_train, y_train, cv=10, scoring=scoring, n_jobs=-1)
    mean_cv_score, std_cv_score = np.mean(cv_scores), np.var(cv_scores)

    bst_eval.fit(X_train, y_train)
    prediction = bst_eval.predict(X_test)
    if task.lower() == 'classification':
        if is_dask:
            eval_score = dask_accuracy_score(y_test, prediction)
        else:
            eval_score = accuracy_score(y_test, prediction)
    else:
        if is_dask:
            eval_score = dask_r2_score(y_test, prediction)
        else:
            eval_score = r2_score(y_test, prediction)

    return mean_cv_score, std_cv_score, eval_score
