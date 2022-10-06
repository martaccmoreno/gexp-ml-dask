from typing import Union

import dask.array as da
import numpy as np

from dask_ml.model_selection import train_test_split as dask_train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from dask_ml.preprocessing import LabelEncoder as DaskLabelEncoder
from dask_ml.preprocessing import StandardScaler as DaskStandardScaler


def pre_ml_processing(feature_array: Union[da.Array, np.array], label_array: Union[da.Array, np.array],
                      is_dask: bool = False, task: str = 'classification') -> \
        Union[tuple[da.Array, da.Array, da.Array, da.Array], tuple[np.array, np.array, np.array, np.array]]:
    """Split processed feature matrix and label vector arrays into train/test sets; then optionally encode the labels;
    lastly standardize the gene expression features.

    Args:
        feature_array (Array): a gene expression matrix array.
        label_array (Array): a label vector array.
        task (string): The machine learning task to perform later. If it's a classification task, encode the labels.
        is_dask (boolean): Whether the Arrays are a lazy Dask Arrays (True) or numpy Arrays (False).

    Returns:
        Four arrays: Corresponding to the train/test splits of the scaled gene expression features and the
        train/test splits of the label arrays, which may optionally be encoded.
    """
    if task.lower() not in ['classification', 'class', 'regression', 'regress']:
        raise ValueError('Invalid ML task!')

    if is_dask:
        X_train, X_test, y_train, y_test = dask_train_test_split(feature_array, label_array, test_size=0.3,
                                                                 shuffle=True, random_state=42)
        y_train, y_test = da.ravel(y_train), da.ravel(y_test)
    else:
        X_train, X_test, y_train, y_test = train_test_split(feature_array, label_array, test_size=0.3,
                                                            shuffle=True, random_state=42)

    if task.lower() == 'classification':
        if is_dask:
            enc = DaskLabelEncoder()
        else:
            enc = LabelEncoder()
        enc.fit(y_train)
        y_train, y_test = enc.transform(y_train), enc.transform(y_test)

    if is_dask:
        sc = DaskStandardScaler()
    else:
        sc = StandardScaler()
    sc.fit(X_train)
    X_train_scaled, X_test_scaled = sc.transform(X_train), sc.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test
