import os
import random

import pandas as pd


def load_gexp_dataset(infolder='tcga-brca-gene-exp', outfile='gene_expression.csv', n_subsamples=None):

    subdirs = os.listdir(infolder)
    if n_subsamples:
        random.seed(42)  # TODO: add as a var
        subdirs = random.choices(subdirs, k=n_subsamples)

    file_paths = []
    for directory in subdirs:
        subdir_path = os.path.join(infolder, directory)
        # Subfolders should each only have one file
        gexp_filename = [f for f in os.listdir(subdir_path) if 'counts' in f.lower() or 'pkm' in f.lower()]
        if gexp_filename:
            gexp_filename = gexp_filename[0]
            gexp_file_path = os.path.join(subdir_path, gexp_filename)
            file_paths.append(gexp_file_path)

    # Obtain column names (for after transposing)
    gene_names = pd.read_csv(file_paths[0], sep='\t', header=None)[0]
    filtered_gene_names = gene_names.loc[gene_names.str.startswith('ENSG')]
    col_names = pd.concat([pd.Series(['filenames']), filtered_gene_names])
    # To write as a row, the series must be converted into a DF and transposed
    pd.DataFrame(col_names).transpose().to_csv(outfile, header=False, index=False)

    for f in file_paths:
        gexp_row = pd.read_csv(f, sep='\t', header=None, index_col=0)
        # Select only rows pertaining to genes
        gexp_row_genes_only = gexp_row.iloc[gexp_row.index.str.startswith('ENSG'), :]
        # Transpose
        transposed_gexp_row = gexp_row_genes_only.transpose()
        # Set filenames as index
        transposed_gexp_row = transposed_gexp_row.assign(filename=os.path.basename(f)).set_index('filename')
        # Append to file
        transposed_gexp_row.to_csv(outfile, mode='a', header=False)
