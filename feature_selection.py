import pandas as pd
import numpy as np
import numpy as np
import argparse
import os
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.linear_model import Lasso

def feature_selection(method, X, Y, k) :
    if method not in ['PCA', 'ANOVA', 'RFE']:
        print("{} not supported.".format(method))
        return

    if(method == "PCA"):
        selector = PCA(n_components=k)
        X_refined = selector.fit_transform(X)
    elif(method == "RFE"):
        selector = RFECV(Lasso(), step=100, cv=5, n_jobs = -1)
        X_refined = selector.fit_transform(X, Y)
    else:
        selector = SelectKBest(f_classif, k=k)
        X_refined= selector.fit_transform(X, Y)

    return X_refined, selector
