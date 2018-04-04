import pandas as pd
import numpy as np
import numpy as np
import argparse
import os
import pickle
import h5py

from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNetCV
from sklearn.svm import LinearSVR, LinearSVC

def feature_selection(X, Y, outcome, method, imp_method, data_dir, verbose=0) :
    if method not in ['RFE', 'PCA', 'ElasticNet']:
        raise Exception("{} not supported.".format(method))

    is_classf = Y.dtype == np.int8
    feature_subset_path = os.path.join(data_dir, 'feature_subset_{}_{}_{}.h5'.format(outcome, method, imp_method))
    if os.path.exists(feature_subset_path):
        if verbose:
            print("Feature subset already exists. Loading {}...".format(feature_subset_path))
        with h5py.File(feature_subset_path, 'r') as hf:
            subset = hf[method][:]
        X_refined = X[:,subset]
        selector = None
    else:   
        if method == 'RFE':
            if is_classf:
                selector = RFECV(LinearSVC(), step=0.1, cv=5, n_jobs=-1, verbose=verbose)
            else:
                selector = RFECV(LinearSVR(), step=0.1, cv=5, n_jobs=-1, verbose=verbose)
            X_refined = selector.fit_transform(X, Y)
        elif method == 'ElasticNet':
            selector = SelectFromModel(ElasticNetCV(cv=10, n_jobs=-1))
            X_refined = selector.fit_transform(X, Y)
        else:
            selector = None
            pca_path = os.path.join(data_dir, 'pca_comp_{}_{}.pkl'.format(outcome, imp_method)) 
            if os.path.exists(pca_path):
                print("PCA components already exist. Loading {}...".format(pca_path))
                pca = joblib.load(pca_path)
                X_refined = pca.transform(X)
            else:
                var_thr = 0.99
                pca = PCA()
                x_pca = pca.fit_transform(X)
                index_pca = np.argmax(pca.explained_variance_ratio_.cumsum() > var_thr)
                if verbose:
                    print("Number of selected features:", index_pca)
                pca = PCA(n_components=index_pca)
                X_refined = pca.fit_transform(X)
                joblib.dump(pca, pca_path)

    if selector:
        with h5py.File(feature_subset_path, 'w') as hf:
            hf.create_dataset(method,  data=selector.get_support())

    return X_refined 

