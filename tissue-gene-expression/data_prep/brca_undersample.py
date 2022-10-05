import dask.dataframe as dd
import pandas as pd
from sklearn.model_selection import train_test_split
import math

brca_fpkm = pd.read_csv('../data/brca_data/brca_fpkm.csv')
brca_subtypes = pd.read_csv('../data/brca_data/brca_subtypes.csv')

n_top_var = 20000
sorted_var = brca_fpkm.var(axis=0).sort_values(ascending=False)
brca_fpkm_top_var = brca_fpkm[sorted_var.iloc[:n_top_var].index]

brca_fpkm_top_var.to_csv(f"../data/brca_undersample/csv/n{brca_fpkm_top_var.shape[0]}_f{n_top_var}_brca_fpkm.csv",
                         index=False)
brca_subtypes.to_csv(f"../data/brca_undersample/csv/n{brca_subtypes.shape[0]}_brca_subtypes.csv", index=False)

# get feature matrix partitions with aprox. 64 MB each
npts = math.ceil(brca_fpkm_top_var.memory_usage(index=True, deep=True).sum()/6.4e7)
brca_fpkm_top_var_dask = dd.from_pandas(brca_fpkm_top_var, npartitions=npts)
brca_subtypes_dask = dd.from_pandas(brca_subtypes, npartitions=1)

brca_fpkm_top_var_dask.to_parquet(f'../data/brca_undersample/parquet/n{brca_fpkm_top_var.shape[0]}_f{n_top_var}_brca_fpkm.parquet',
                                  engine='pyarrow', compression='snappy')
brca_subtypes_dask.to_parquet(f"../data/brca_undersample/parquet/n{brca_subtypes.shape[0]}_brca_subtypes.parquet",
                              engine='pyarrow', compression='snappy')

num_subsamples = [200, 600]
for nsub in num_subsamples:
    _, subsample_brca_fpkm, _, subsample_brca_subtypes = train_test_split(brca_fpkm_top_var, brca_subtypes,
                                                                          test_size=nsub, stratify=brca_subtypes,
                                                                          random_state=42)

    subsample_brca_fpkm.to_csv(f"../data/brca_undersample/csv/brca_undersample/csv/n{nsub}_f{n_top_var}_brca_fpkm.csv", index=False)
    subsample_brca_subtypes.to_csv(f"../data/brca_undersample/csv/brca_undersample/csv/n{nsub}_brca_subtypes.csv", index=False)

    npts = math.ceil(subsample_brca_fpkm.memory_usage(index=True, deep=True).sum()/6.4e7)
    dd.to_parquet(dd.from_pandas(subsample_brca_fpkm, npartitions=npts),
                  f"..data/brca_undersample/parquet/n{nsub}_f{n_top_var}_brca_fpkm.parquet",
                  engine='pyarrow', compression='snappy')
    dd.to_parquet(dd.from_pandas(subsample_brca_subtypes, npartitions=1),
                  f"..data/brca_undersample/parquet/n{nsub}_brca_subtypes.parquet",
                  engine='pyarrow', compression='snappy')


num_features = [10000, 40000]
sorted_var = brca_fpkm.var(axis=0).sort_values(ascending=False)
for nfeat in num_features:
    feat_sel_brca_fpkm = brca_fpkm[sorted_var.iloc[:nfeat].index]

    feat_sel_brca_fpkm.to_csv(f"..data/brca_undersample/csv/n{brca_fpkm.shape[0]}_f{nfeat}_brca_fpkm.csv", index=False)
    dd.to_parquet(dd.from_pandas(feat_sel_brca_fpkm, npartitions=1),
                  f"..data/brca_undersample/parquet/n{brca_fpkm.shape[0]}_f{nfeat}_brca_fpkm.parquet",
                  engine='pyarrow', compression='snappy')