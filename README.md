# Scalable Transcriptomics Analysis with Dask

The present repository contains scripts and datasets that support the paper “Scalable Transcriptomics Analysis with Dask: Applications in Transcriptomics and Data Science” by Marta Moreno, Ricardo Vilaça and Pedro G. Ferreira, currently under review.

This work illustrates how Dask can boost genomic data science and machine learning applications. We also review the main steps behind scalable data science.

We analysesd two types of transcriptomics data:

1. Gene expression levels sampled from cancer tissues and a distributions mimicking gene counts (tissue-gene-expression);
2. Gene expression levels measured from single cell RNA-seq (scRNA-seq) data (single-cell-gene-expression).

In 1), several tests involving end-to-end machine learning pipelines were performed using both Dask and Scikit-learn. 
In 2), a three-step scRNA-seq processing pipeline was applied to the data.

A companion step-by-step guide that explains how a default version of the typical machine learning pipeline for predicting molecular subtypes and other classes, as seen in 1), is hosted on Binder:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/martaccmoreno/gexp-ml-tutorial/HEAD)
