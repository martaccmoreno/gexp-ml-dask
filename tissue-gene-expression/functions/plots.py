import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def plot_gexp_counts(gene_expression_array):
    sns.set_theme(rc={'figure.figsize': (28, 16)}, style='whitegrid', font_scale=2.5,
                  palette='crest')

    mean_expression_per_gene = gene_expression_array.mean()
    if type(mean_expression_per_gene) != np.float64:
        gexp_values = mean_expression_per_gene.values
        sns.histplot(x=gexp_values)
    else:
        plt.hist(x=mean_expression_per_gene)

    plt.title("Frequency Distribution of Gene Expression Levels\n")
    plt.xlabel("Gene Expression Levels")
    plt.ylabel("Frequency")
    plt.show()
    plt.close('all')


def plot_log_transform(gene_expression_array, linear_regress_line=True):
    """
    Plot difference caused by log transform on a gene expression array.
    """

    sns.set_theme(rc={'figure.figsize': (32, 16)}, style='whitegrid', font_scale=2.5,
                  palette='crest')

    # only necessary for dask
    # gene_expression_array = gene_expression_array.compute().astype(np.float)

    mean_expression_per_gene = gene_expression_array.mean()

    if linear_regress_line:
        uniform = list(range(mean_expression_per_gene.shape[0]))
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(uniform, mean_expression_per_gene)
        plt.plot(uniform, [slope*x+intercept for x in uniform], c='black', linewidth=2)

    # x: index, the gene names; y: values, the mean gene expression values per gene

    plt.title("Distribution of mean expression per gene before log10 transformation\n")
    plt.xlabel('Genes')
    plt.ylabel('FPKM')
    plt.xticks([])
    plt.show()
    plt.close('all')

    log_transformed_mean_expression_per_gene = np.log10(mean_expression_per_gene + 1.0)

    if linear_regress_line:
        uniform = list(range(log_transformed_mean_expression_per_gene.shape[0]))
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(uniform,
                                                                             log_transformed_mean_expression_per_gene)
        plt.plot(uniform, [slope*x+intercept for x in uniform], c='black', linewidth=2)

    sns.scatterplot(x=log_transformed_mean_expression_per_gene.index,
                    y=log_transformed_mean_expression_per_gene.values, s=50)
    plt.title("Distribution of mean expression per gene after log10 transformation\n")
    plt.xlabel('Genes')
    plt.ylabel('log10 + 1-scaled FPKM')
    plt.xticks([])
    # plt.show()
    plt.savefig('confusion_matrix.png')
    plt.close('all')


def plot_confusion_matrix(estimator, y_test, y_pred, enc, normalize=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    sns.set_theme(rc={'figure.figsize': (12, 12)}, style='whitegrid', font_scale=1.5)

    cm = confusion_matrix(y_test, y_pred)

    cm_title = "Confusion matrix for " + str(estimator).split('(')[0]
    if normalize:
        cm_title += ' with normalization'

    plt.figure(figsize=(10, 10))
    heatmap = sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    heatmap.set_title(cm_title)
    heatmap.yaxis.set_ticklabels(enc.classes_, rotation=0, ha='right')
    heatmap.xaxis.set_ticklabels(enc.classes_, rotation=45, ha='right')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    # plt.savefig('confusion_matrix.png')
    plt.close('all')


def plot_class_report(estimator, y_test, y_pred, enc):
    sns.set_theme(rc={'figure.figsize': (18, 10)}, style='whitegrid', font_scale=1.5,
                  palette='crest')

    class_report = classification_report(y_test, y_pred, output_dict=True,
                                         labels=np.unique(np.concatenate([y_test, y_pred])),
                                         target_names=enc.classes_)
    estimator_name = str(estimator).split('(')[0]

    # Check colors for BRCA subtypes in paper
    per_class_df = pd.DataFrame.from_dict(class_report)[enc.classes_].loc[["precision", "recall", "f1-score"], :]
    per_class_df.plot(kind='bar', color=["mediumpurple", "indianred", "limegreen", "deepskyblue", "pink"])

    plt.ylim([0, 1.0])
    plt.tick_params('x', rotation=45)
    plt.tick_params('y')
    plt.title(f"Test metrics for BRCA molecular subtypes predicted with {estimator_name}\n")
    plt.legend()
    plt.show()
    # plt.savefig('class_report.png')
    plt.close()
