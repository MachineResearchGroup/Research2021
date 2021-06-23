import pandas as pd
import warnings
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.csr import csr_matrix


def print_results_cv(results, indent=0):
    for key, value in results.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            print_results_cv(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])


def get_general_cv_mean(resampling):
    return pd.read_csv('../results/evaluation/general_cv_mean(' + resampling +').csv')


def get_final_mean(resampling):
    return pd.read_csv('../results/evaluation/final_mean(' + resampling +').csv')

