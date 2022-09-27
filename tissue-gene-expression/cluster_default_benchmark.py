import csv
import gc
import os
import timeit

import dask.array as da
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import lightgbm as lgb

from dask.distributed import Client
from dask_jobqueue import SLURMCluster
from dask_ml.model_selection import train_test_split as dask_train_test_split
from dask_ml.preprocessing import LabelEncoder as DaskLabelEncoder
from dask_ml.preprocessing import StandardScaler as DaskStandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class UpperQuartile(BaseEstimator, TransformerMixin):
    """
    This estimator learns a normalization factor from the data's upper quartile q,
    and uses it as a basis for the scaling factor.

    Note that UpperQuartile assumes all samples have nonzero transcripts.
    """

    def __init__(self, q=0.75):
        self.q = q

    def fit(self, X):
        # Remove all transcripts that are 0 across ALL samples, i.e. their per-gene mean is equal to 0
        self.X = X.loc[:, (X.mean(axis=0) > 0.0)]
        self.norm_factor = self._uq(self.X)
        # For symmetry, normalization factors are adjusted to multiply to 1 before being used as scaling factors.
        self.scaling_factor = self.norm_factor / np.exp(np.mean(np.log(self.norm_factor.replace(0, 1).values)))
        return self

    def _uq(self, X):
        return X.apply(lambda sample: sample.quantile(self.q) / sample.sum(), axis=1)

    def transform(self, X):
        return X.multiply(self.scaling_factor, axis=0)


class UpperQuartileDask(BaseEstimator, TransformerMixin):
    """
    This estimator learns a normalization factor from the data's upper quartile q,
    and uses it as a basis for the scaling factor.

    Note that UpperQuartile assumes all samples have nonzero transcripts.
    """

    def __init__(self, q=0.75):
        self.q = q

    def fit(self, X):
        # Remove all transcripts that are 0 across ALL samples, i.e. their per-gene mean is equal to 0
        self.X = X.map_partitions(lambda df: df[df.columns[df.mean(axis=0) > 0]])
        self.norm_factor = self._uq(self.X)
        # For symmetry, normalization factors are adjusted to multiply to 1 before being used as scaling factors.
        self.scaling_factor = self.norm_factor / da.exp(da.mean(da.log(self.norm_factor.replace(0, 1).values)))
        return self

    def _uq(self, X):
        return X.map_partitions(lambda df: df.apply(lambda sample: sample.quantile(self.q) / sample.sum(), axis=1))

    def transform(self, X):
        return X.mul(self.scaling_factor, axis=0)


def dask_load_data(filepaths):
    print("\nLoad feature matrix")
    if 'parquet' in filepaths[0]:
        dask_feature_matrix = dd.read_parquet(filepaths[0]).persist()
    else:
        print("from csv")
        dask_feature_matrix = dd.read_csv(filepaths[0], assume_missing=True, sample=2000000)
        # It is important for feature and label divisions to match for later train/test splitting.
    if 'parquet' in filepaths[1]:
        dask_label_vector = dd.read_parquet(filepaths[1])
    else:
        dask_label_vector = dd.read_csv(filepaths[1])

    return dask_feature_matrix, dask_label_vector


def dask_feature_preprocessing(feature_matrix: dd.DataFrame) -> dd.DataFrame:
    uq = UpperQuartileDask()
    uq_feature_matrix = uq.fit_transform(feature_matrix).persist()
    del feature_matrix
    gc.collect()

    # For gene features to be maximally informative, we want them to have a minimum expression signal and to
    # vary at least a little across samples.
    mean = uq_feature_matrix.mean(axis=0).persist()
    var = uq_feature_matrix.var(axis=0).persist()

    threshold_feature_matrix = uq_feature_matrix[uq_feature_matrix.columns[(mean > mean.quantile(0.25)) &
                                                                           (var > var.quantile(0.25))]
    ].repartition(partition_size='64MB')

    log_scaled_feature_matrix = threshold_feature_matrix.applymap(lambda gene: da.log2(gene + 1))

    return log_scaled_feature_matrix


def df_to_array(preprocessed_feature_matrix: dd.DataFrame, dask_label_vector: dd.Series):
    preprocessed_feature_array = preprocessed_feature_matrix.to_dask_array(lengths=True).rechunk('auto')
    label_array = dask_label_vector.to_dask_array(lengths=True).rechunk((preprocessed_feature_array.chunks[0], None))

    return preprocessed_feature_array, label_array


def pre_ml_processing(dask_feature_array: da.Array, dask_label_array: da.Array):
    X_train, X_test, y_train, y_test = dask_train_test_split(dask_feature_array, dask_label_array, test_size=0.3,
                                                        shuffle=True, random_state=42)
    y_train, y_test = da.ravel(y_train), da.ravel(y_test)

    # Persists are very important here because we use these X and y training/test arrays a lot.
    enc = DaskLabelEncoder()
    enc.fit(y_train)
    y_train_persisted, y_test_persisted = enc.transform(y_train).persist(), enc.transform(y_test).persist()

    print("Standard scaling...\n")
    sc = DaskStandardScaler()
    sc.fit(X_train)
    X_train_scaled, X_test_scaled = sc.transform(X_train).persist(), sc.transform(X_test).persist()

    return X_train_scaled, X_test_scaled, y_train_persisted, y_test_persisted


def dask_default_lgbm_pipeline(X_train: da.Array, X_test: da.Array, y_train: da.Array, y_test: da.Array):

    lgbm_eval = lgb.DaskLGBMClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False, n_jobs=-1)
    lgbm_eval.fit(X_train, y_train)
    prediction = lgbm_eval.predict(X_test)
    eval_score = accuracy_score(y_test, prediction)
    return eval_score


def dask_pipeline(filepaths):
    print("\nLoad data...")
    dask_feature_matrix, dask_label_vector = dask_load_data(filepaths)
    dask_feature_matrix = dask_feature_matrix.persist()

    print("\nPreprocess features...")
    preprocessed_feature_matrix = dask_feature_preprocessing(dask_feature_matrix)
    del dask_feature_matrix
    gc.collect()

    print("\nConvert to dask array...")
    # For proper splitting into train/test sets, the chunk sizes must align for the two datasets.
    preprocessed_feature_array, label_array = df_to_array(preprocessed_feature_matrix, dask_label_vector)
    del dask_label_vector
    gc.collect()

    print("\nPre-ml processing...")
    X_train, X_test, y_train, y_test = pre_ml_processing(preprocessed_feature_array, label_array)
    del preprocessed_feature_array, label_array
    gc.collect()

    print("\nDefault ML pipeline...")
    eval_score = dask_default_lgbm_pipeline(X_train, X_test, y_train, y_test)

    return eval_score
#####

def nondask_load_data(filepaths):
    return pd.read_csv(filepaths[0], index_col=0).reset_index(), pd.read_csv(filepaths[1], index_col=0).reset_index()


def nondask_feature_preprocessing(feature_matrix):
    uq = UpperQuartile()
    uq_feature_matrix = uq.fit_transform(feature_matrix)

    # For gene features to be maximally informative, we want them to have a minimum expression signal and to
    # vary at least a little across samples.
    mean = feature_matrix.mean(axis=0)
    var = feature_matrix.var(axis=0)

    threshold_feature_matrix = uq_feature_matrix[uq_feature_matrix.columns[(mean > mean.quantile(0.25)) &
                                                                           (var > var.quantile(0.25))]]

    log_scaled_feature_matrix = threshold_feature_matrix.applymap(lambda gene: np.log2(gene + 1))

    return log_scaled_feature_matrix


def nondask_pre_ml_processing(nondask_feature_array, nondask_label_array):

    X_train, X_test, y_train, y_test = train_test_split(nondask_feature_array, nondask_label_array, test_size=0.3,
                                                        shuffle=True, random_state=42)

    enc = LabelEncoder()
    enc.fit(y_train)
    y_train_encoded, y_test_encoded = enc.transform(y_train), enc.transform(y_test)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_scaled, X_test_scaled = sc.transform(X_train), sc.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded


def nondask_lgbm_default_pipeline(X_train, X_test, y_train, y_test):
    bst = lgb.LGBMClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
    bst.fit(X_train, y_train)
    prediction = bst.predict(X_test)
    eval_score = accuracy_score(y_test, prediction)

    return eval_score


def nondask_pipeline(filepaths: tuple=None):
    nondask_feature_matrix, nondask_label_vector = nondask_load_data(filepaths)

    preprocessed_feature_matrix = nondask_feature_preprocessing(nondask_feature_matrix)
    gc.collect()

    X_train, X_test, y_train, y_test = nondask_pre_ml_processing(preprocessed_feature_matrix, nondask_label_vector)
    y_train, y_test = np.ravel(y_train), np.ravel(y_test)
    del preprocessed_feature_matrix, nondask_label_vector
    gc.collect()

    eval_score = nondask_lgbm_default_pipeline(X_train, X_test, y_train, y_test)

    return eval_score


if __name__ == '__main__':
    fps_dask = [
        ('n5000_f20000_synthetic_features.parquet',
         'n5000_synthetic_labels.csv')
    ]

    fps_nondask = [
        ('n5000_f20000_synthetic_features.csv',
         'n5000_synthetic_labels.csv')
    ]

    runtime = {}
    eval_scores = {}

    for fp in fps_nondask:
        dimensions = f"{os.path.basename(fp[0]).split('_')[0]}x{os.path.basename(fp[0]).split('_')[1]}"
        print(dimensions)

        nondask_memory = None
        retval = nondask_pipeline(fp)
        mean_cv_score, std_cv_score, eval_score = retval

        nondask_runtime = min(timeit.repeat("nondask_pipeline(fp)",
                                            setup="from __main__ import nondask_pipeline, fp", repeat=3, number=1))

        runtime[dimensions] = {'Scientific Python Environment': nondask_runtime}
        eval_scores[dimensions] = {'Scientific Python Environment': eval_score}
        gc.collect()

    print("With SPE results")
    print(runtime)
    print(eval_scores)

    n_nodes = [2, 4, 8, 32, 64]  # adjust as needed
    # comment spe code above to run one at a time if walltime is an issue
    n_processes = 16  # adjust n processes as needed
    for nd in n_nodes:
        print(f"Number Nodes = {nd}")
        cluster = SLURMCluster(
            job_extra=["--qos=debug"],
            cores=16,
            processes=n_processes,
            memory="25GB",
            local_directory="/",
            interface="ib0",
            project="acct",  # associated slurm account
            queue="debug",
            walltime="6:00:00",  # play with lower walltimes to get ahead in the queue
            env_extra=['export LANG="en_US.utf8"',
                       'export LANGUAGE="en_US.utf8"',
                       'export LC_ALL="en_US.utf8"']
        )
        cluster.scale(jobs=nd)
        client = Client(cluster)
        client.wait_for_workers(n_workers=nd)

        for fp in fps_dask:
            dimensions = f"{os.path.basename(fp[0]).split('_')[0]}x{os.path.basename(fp[0]).split('_')[1]}"

            retval = dask_pipeline(fp)
            mean_cv_score, std_cv_score, eval_score = retval
 
            dask_distributed_runtime = min(timeit.repeat("dask_pipeline(fp)",
                                                         setup="from __main__ import dask_pipeline, client, fp",
                                                         repeat=3, number=1))

            runtime[dimensions].update({'Dask (n = {nd})': dask_distributed_runtime})
            eval_scores[dimensions].update({'Dask (n = {nd})': eval_score})
            gc.collect()

            print(f"After adding Dask N={nd} results")
            print(runtime)
            print(eval_scores)

        client.close()
        cluster.close()

    with open("runtime_benchmark.csv", "w") as outfile:
        w_time = csv.writer(outfile)
        w_time.writerow(['Dimensions', 'Framework', "Fastest Runtime (s)"])
        for key, val in runtime.items():
            for subkey, subval in val.items():
                w_time.writerow([key, subkey, subval])

    pd.DataFrame(runtime).transpose().plot(kind='bar')
    plt.xticks(rotation=360)
    plt.ylabel('Minimum Runtime (s)')
    plt.xlabel('Framework')
    plt.savefig('runtime_benchmark.pdf')
