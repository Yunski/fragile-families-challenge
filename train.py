import argparse
import numpy as np
import pandas as pd
import os
import scipy
import time

from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, LogisticRegressionCV
from sklearn.metrics import brier_score_loss, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import resample

from feature_selection import feature_selection
from utils import get_data

def train(classifier, X, Y, is_classf, cv=10):
    manual_cv = False
    if classifier == 'ElasticNet':
        if is_classf:
            raise Exception("Choose another model for classification")
        model = ElasticNetCV(cv=cv)
    elif classifier == 'Lasso':
        if is_classf:
            raise Exception("Choose another model for classification")
        model = LassoCV(cv=cv)
    elif classifier == 'AdaBoost':
        estimator = DecisionTreeClassifier(max_depth=1) if is_classf else DecisionTreeRegressor(max_depth=1)
        n = 100
        model = AdaBoostClassifier(estimator, n_estimators=100) if is_classf else AdaBoostRegressor(estimator, n_estimators=100)
        manual_cv = True
    else:
        raise ValueError("model {} not available".format(classifier))

    metric = brier_score_loss if is_classf else mean_squared_error
    metric_str = 'brier_loss' if is_classf else 'mse'
    print("Training {}...".format(classifier))
    print("10-Fold cross validation...")
    start = time.time()
    if manual_cv:
        kf = KFold(n_splits=cv)
        kf.get_n_splits(X)
        losses = []
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            print("Fold {}".format(i))
            X_train, Y_train = X[train_index], Y[train_index]
            X_test, Y_test = X[test_index], Y[test_index] 
            model.fit(X_train, Y_train)
            Y_pred = model.predict(X_test)
            losses.append(metric(Y_test, Y_pred))
        mean_loss = np.mean(losses)
    else:
        model.fit(X, Y)
        Y_pred = model.predict(X)
        mean_loss_by_alpha = np.mean(model.mse_path_, axis=1)
        mean_loss = np.min(mean_loss_by_alpha)
    print("Training took {:.4f}s.".format(time.time()-start))
    print("cv mean {}: {:.4f}".format(metric_str, mean_loss))

    print("Bootstrapping...B)")
    n = len(X)
    bs_losses = []
    for i in range(10):
        print("Sample {}".format(i))
        data = np.hstack((X, Y.reshape(len(Y), 1)))
        train = resample(data, n_samples=int(0.9*n))
        test = np.array([sample for sample in data if sample.tolist() not in train.tolist()])
        X_train, Y_train = train[:,:-1], train[:,-1]
        X_test, Y_test = test[:,:-1], test[:,-1]
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        bs_losses.append(metric(Y_test, Y_pred))
    mean_loss = np.mean(bs_losses)
    print("bootstrap mean {}: {:.4f}".format(metric_str, mean_loss))
    alpha = 0.95
    p = ((1.0-alpha)/2.0) * 100
    lower = max(0.0, np.percentile(bs_losses, p))
    p = (alpha+((1.0-alpha)/2.0)) * 100
    upper = min(1.0, np.percentile(bs_losses, p))
    print("95% confidence interval: [{:.4f}, {:.4f}]".format(lower, upper))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sentiment Analysis")
    parser.add_argument('model', help="model")
    parser.add_argument('X', help="features csv filename")
    parser.add_argument('Y', help="labels csv filename")
    parser.add_argument('-m', dest='fs_method', help="feature selection method", default='PCA')
    parser.add_argument('-k', dest='k', help="k", type=int, default='100')
    parser.add_argument('-d', dest='data_dir', help='data directory', default='data')
    args = parser.parse_args()

    print("Loading dataset...")
    X, Y = get_data(args.data_dir, args.X, args.Y)
    is_classf = Y.dtype == np.int8 
    print("Successfully loaded dataset.")
    print("Performing feature selection using {}...".format(args.fs_method))
    X, selector = feature_selection(args.fs_method, X, Y, args.k)
    print("X dim: {}".format(X.shape))
    train(args.model, X, Y, is_classf)

