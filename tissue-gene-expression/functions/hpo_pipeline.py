from typing import Union

import dask
import dask.array as da
import numpy as np
import xgboost as xgb

from dask_ml.model_selection import GridSearchCV as GridSearchCVDask
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from dask_ml.metrics import accuracy_score as dask_accuracy_score

from dask_cv import dask_cv


def hpo_pipeline(param_dist: dict, X_train: Union[da.Array, np.array], X_test: Union[da.Array, np.array],
                 y_train: Union[da.Array, np.array], y_test: Union[da.Array, np.array],
                 is_dask: bool = False):
    """Pipeline for passing prepared feature and label train/test sets through a Stochastic Gradient Descent estimator.
    Works only for classification.

    Args:
    X_train, X_test: train and test gene expression feature arrays.
    y_train, y_test: train and test label arrays.
    is_dask (boolean): Whether the Arrays are a lazy Dask Arrays (True) or numpy Arrays (False).

    Returns:
    Mean Cross-Validation (CV) Score (float)
    Standard Deviation for the CV Score (float)
    Evaluation Score for the model on the test set (float)
    """
    # Dask's RandomizedSearch implementation only accepts sckit-learn estimators
    bst_cv = xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1, use_label_encoder=False)

    if is_dask:
        cv_search_sgd = GridSearchCVDask(bst_cv, param_dist, scoring='accuracy', cv=2,
                                         n_jobs=-1)
        cv_scores = dask_cv(cv_search_sgd, X_train, y_train, cv_splits=5, hpo=True)
        cv_scores = dask.compute(cv_scores)
    else:
        cv_search_sgd = GridSearchCV(bst_cv, param_dist, scoring='accuracy', cv=2, n_jobs=-1)
        cv_scores = cross_val_score(cv_search_sgd, X_train, y_train, cv=5, n_jobs=-1)
    mean_cv_score, std_cv_score = np.mean(cv_scores), np.var(cv_scores)

    bst_eval = xgb.XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1, use_label_encoder=False)
    if is_dask:
        search_sgd = GridSearchCVDask(bst_eval, param_dist, scoring='accuracy', cv=2, n_jobs=-1)
        X_test = X_test.compute()
    else:
        search_sgd = GridSearchCV(bst_eval, param_dist, scoring='accuracy', cv=2, n_jobs=-1)

    search_sgd.fit(X_train, y_train)
    prediction = search_sgd.best_estimator_.predict(X_test)
    if is_dask:
        eval_score = dask_accuracy_score(y_test, prediction)
    else:
        eval_score = accuracy_score(y_test, prediction)

    return mean_cv_score, std_cv_score, eval_score
