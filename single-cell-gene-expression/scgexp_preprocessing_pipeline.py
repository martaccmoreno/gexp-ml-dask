import csv
import gc
import timeit

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import dask.dataframe as dd
import dask.array as da
from dask.distributed import Client


def load_data(file):
    df = pd.read_parquet(file)
    return df


def lognormalize(df):
    col_sums = df.sum(axis=0).replace(0, 0.000001)
    scaled_by_col = df / col_sums  # we cannot divide by 0
    # multiply by factor of 10,000
    multiplied_scaled_by_col = scaled_by_col * 10000
    # log-scaling 0 results in -inf so we replace that by 1, which results in 0
    multiplied_scaled_by_col = multiplied_scaled_by_col.replace(0, 1)
    # irrelevant if axis is 0 or 1, but dask's apply only works with axis=1
    log_normalized = multiplied_scaled_by_col.apply(np.log, axis=1)

    return log_normalized


# pipeline based off: https://satijalab.org/seurat/articles/pbmc3k_tutorial.html#normalizing-the-data-1
def pipeline(file):
    df = load_data(file)
    lognorm = lognormalize(df)
    highly_var_genes = lognorm.var(axis=0).nlargest(n=2000).index
    lognorm_feat_sel = lognorm.loc[:, highly_var_genes]

    # Scaling the data
    # Shifts the expression of each gene, so that the mean expression across cells is 0 and the variance is 1
    gene_mean = lognorm_feat_sel.mean(axis=0)
    gene_var = lognorm_feat_sel.var(axis=0)
    expr_scaled = (lognorm_feat_sel - gene_mean) / gene_var
    print(expr_scaled.head())

    return expr_scaled


def load_data_dask(file):
    dask_df = dd.read_parquet(file)
    return dask_df


def lognormalize_dask(df_dask):
    col_sums = df_dask.sum(axis=0)
    scaled_by_col = df_dask / col_sums.replace(0, 0.000001)   # we cannot divide by 0
    # log-scaling 0 results in -inf so we replace that by 1, which results in 0
    multiplied_scaled_by_col = scaled_by_col.mul(10000, axis=1).replace(0, 1)
    log_normalized = multiplied_scaled_by_col.apply(da.log, axis=1)

    return log_normalized


def feat_scaling_dask(df_dask):
    gene_mean = df_dask.mean(axis=0)
    gene_var = df_dask.var(axis=0)
    expr_scaled = (df_dask - gene_mean) / gene_var
    return expr_scaled


def pipeline_dask(file):
    df_dask = load_data_dask(file)

    # Map partition helps further parallelize computation and simplifies Dask graph building by reducing n tasks
    lognorm = dd.map_partitions(lognormalize_dask, df_dask, meta=df_dask)

    # However map partitions cannot be used in all cases. For example, here it wouldn't work because
    # it applies a function to each partition (pandas dfs inside dask df) and we have a selection,
    # which must apply to the entire dataset (all partitions).
    highly_var_genes = lognorm.var(axis=0).nlargest(n=2000).index.compute()
    lognorm_feat_sel = lognorm.loc[:, highly_var_genes]
    # Persist is useful when we significantly reduce dataset dimensions but might increase memory usage
    lognorm_feat_sel_persisted = lognorm_feat_sel.persist()

    # Scaling the data
    # Shifts the expression of each gene, so that the mean expression across cells is 0 and the variance is 1
    expr_scaled = dd.map_partitions(feat_scaling_dask, lognorm_feat_sel_persisted, meta=lognorm_feat_sel_persisted)
    expr_scaled_computed = expr_scaled.compute()
    print(expr_scaled_computed.head())

    return expr_scaled_computed


if __name__ == '__main__':
    dims = {10000: '10k', 20000: '20k', 40000: '40k', 87947: '87.9k',
            2500: '2.5k', 5000: '5k', 24245: '24.2k', 60000: '60k', 15000: '15k'}
    dims_small = ['5kx5k.parquet', '2.5kx10k.parquet', '10kx2.5k.parquet']
    dims_large = ['10kx10k.parquet', '5kx20k.parquet', '20kx5k.parquet']
    dim_runtime_compare = {'n40k x f10k': {}, 'n60k x f10k': {}, 'n87.9k x f10k': {}, 'n40k x f15k': {},
                           'n60k x f15k': {}, 'n87.9k x f15k': {}, 'n40k x f20k': {}, 'n60k x f20k': {},
                           'n87.9k x f20k': {}, 'n40k x f24.2k': {}, 'n60k x f24.2k': {}, 'n87.9k x f24.2k': {}}

    for f in [10000, 15000, 20000, 24245]:
        for n in [40000, 60000, 87947]:
            for n_workers, fw in [(1, 'Distributed Threads'), (8, 'Distributed Processes')]:

                file_dask = f'n{n}xf{f}.parquet'
                key = f'n{dims[n]} x f{dims[f]}'
                print(key)

                client = Client(n_workers=n_workers)
                print(client)
                min_dask = min(timeit.repeat("pipeline_dask(file_dask)",
                                             setup="from __main__ import pipeline_dask, file_dask", repeat=3, number=1))
                dim_runtime_compare[key] = {fw: min_dask}
                print(dim_runtime_compare)
                client.wait_for_workers()
                client.close()
                gc.collect()

            with open(f"sc_runtime.csv", "w") as outfile:
                w_time = csv.writer(outfile)
                w_time.writerow(['Dimension', 'Framework', "Minimum Runtime (s)"])
                for key, val in dim_runtime_compare.items():
                    for subkey, subval in val.items():
                        w_time.writerow([key, subkey, subval])

            pd.DataFrame(dim_runtime_compare).transpose().plot(kind='barh')
            plt.title('Runtime for different sampled SC datasets')
            plt.xticks(rotation=90)
            plt.ylabel('Minimum Runtime (s)')
            plt.xlabel('Dataset Dimensions')
            plt.tight_layout()
            plt.savefig(f'sc_runtime.pdf')

    # Some of these dimensions might blow up memory and lead to sigkill even on a machine with 16GB of RAM and
    # 16 GB of additional swap memory.
    for f in [10000, 15000, 20000, 24245]:
        for n in [40000, 60000, 87947]:
            file_spe = f'n{n}xf{f}_whole.parquet'
            key = f'n{dims[n]} x f{dims[f]}'
            print(key)

            min_spe = min(timeit.repeat("pipeline(file_spe)",
                                        setup="from __main__ import pipeline, file_spe", repeat=3, number=1))
            gc.collect()
            dim_runtime_compare[key].update({'SPE': min_spe})
            print(dim_runtime_compare)

            with open(f"sc_runtime.csv", "w") as outfile:
                w_time = csv.writer(outfile)
                w_time.writerow(['Dimension', 'Framework', "Minimum Runtime (s)"])
                for key, val in dim_runtime_compare.items():
                    for subkey, subval in val.items():
                        w_time.writerow([key, subkey, subval])

            pd.DataFrame(dim_runtime_compare).transpose().plot(kind='bar')
            plt.title('Runtime for different sampled SC datasets')
            plt.xticks(rotation=90)
            plt.ylabel('Minimum Runtime (s)')
            plt.xlabel('Dataset Dimensions')
            plt.tight_layout()
            plt.savefig(f'sc_runtime.pdf')
