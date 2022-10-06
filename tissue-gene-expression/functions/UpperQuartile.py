import numpy as np
import dask.array as da
from sklearn.base import BaseEstimator, TransformerMixin

# Adapted from: https://rdrr.io/bioc/edgeR/src/R/calcNormFactors.R
# Verified using the data in: https://davetang.org/muse/2011/01/24/normalisation-methods-for-dge-data/


class UpperQuartile(BaseEstimator, TransformerMixin):
    """This estimator learns a normalization factor from the data's upper quartile q, and uses it as a basis for the
    scaling factor. Note that UpperQuartile assumes all samples have nonzero transcripts.

    Args:
        is_dask (boolean): Whether the Dataframe is represented as a lazy Dask task graph or is already fully loaded
        into memory.

    Returns:
        A custom scikit-learn estimator that learns from and transforms tabular data.
    """

    def __init__(self, q=0.75, is_dask=False):
        self.q = q
        self.is_dask = is_dask
        self.X = None
        self.norm_factor = None
        self.scaling_factor = None

    def fit(self, X):
        # Remove all transcripts that are 0 across ALL samples, i.e. their per-gene mean is equal to 0
        if self.is_dask:
            self.X = X[X.columns[(X.mean(axis=0) > 0.0)]].persist()
            self.norm_factor = self._uq(self.X).persist()
        else:
            self.X = X.loc[:, (X.mean(axis=0) > 0.0)]
            self.norm_factor = self._uq(self.X)
        # For symmetry, normalization factors are adjusted to multiply to 1 before being used as scaling factors.
        if self.is_dask:
            self.scaling_factor = self.norm_factor / da.exp(da.mean(da.log(self.norm_factor.replace(0, 1).values)))
        else:
            self.scaling_factor = self.norm_factor / np.exp(np.mean(np.log(self.norm_factor.replace(0, 1).values)))
        return self

    def _uq(self, X):
        if self.is_dask:
            return X.apply(lambda sample: sample.quantile(self.q) / sample.sum(), axis=1,
                           meta=('UQ', 'f8'))
        else:
            return X.apply(lambda sample: sample.quantile(self.q) / sample.sum(), axis=1)

    def transform(self, X):
        return X.mul(self.scaling_factor, axis=0)
