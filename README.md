# Scalable Transcriptomics Analysis with Dask

The present repository contains all scripts used to generate the results shown in the paper “Scalable Transcriptomics Analysis with Dask: Applications in Transcriptomics and Data Science”. We analysesd two types of transcriptomics data:

1. Gene expression levels sampled from cancer tissues and a distributions mimicking gene counts (tissue-gene-expression);
2. Gene expression levels measured from single cell RNA-seq (scRNA-seq) data (single-cell-gene-expression).

In 1), several tests involving end-to-end machine learning pipelines were performed using both Dask and scikit-learn. In 2), a three-step scRNA-seq processing pipeline was applied to the data.

A companion step-by-step guide that explains how a default version of the typical machine learning pipeline for predicting molecular subtypes and other classes, as seen in 1), is hosted on Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/martaccmoreno/gexp-ml-tutorial/HEAD)
