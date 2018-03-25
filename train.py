import argparse
import numpy as np
import pandas as pd
import os
import scipy
import time

from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.metrics import brier_score_loss, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVR, LinearSVC 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import resample

from feature_selection import feature_selection
from utils import get_data

def train(classifier, X, Y, is_classf, cv=10):
    if classifier == 'ElasticNet':
        if is_classf:
            model = SGDClassifier(loss='log', penalty='elasticnet')
        else:
            model = SGDRegressor(penalty='elasticnet')
    elif classifier == 'Lasso':
        if is_classf:
            model = SGDClassifier(loss='log', penalty='l1')
        else:
            model = SGDClassifier(penalty='l1')
    elif classifier == 'AdaBoost':
        if is_classf:
            estimator = DecisionTreeClassifier(max_depth=1) 
            model = AdaBoostClassifier(estimator, n_estimators=100) 
        else: 
            estimator = DecisionTreeRegressor(max_depth=1)
            model = AdaBoostRegressor(estimator, n_estimators=100)
    elif classifier == 'GP':
        if is_classf:
            model = GaussianProcessClassifier()
        else:
            model = GaussianProcessRegressor() 
    elif classifier == 'SVM':
        if is_classf:
            model = LinearSVC()
        else:
            model = LinearSVR()
    else:
        raise ValueError("model {} not available".format(classifier))

    metric = brier_score_loss if is_classf else mean_squared_error
    metric_str = 'brier_loss' if is_classf else 'mse'
    print("Training {}...".format(classifier))
    print("10-Fold cross validation...")
    start = time.time()

    kf = KFold(n_splits=cv)
    kf.get_n_splits(X)
    losses = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        print("Fold {}".format(i+1))
        X_train, Y_train = X[train_index], Y[train_index]
        X_test, Y_test = X[test_index], Y[test_index] 
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        losses.append(metric(Y_test, Y_pred))
    mean_loss = np.mean(losses)

    total = int(time.time()-start)
    print("Training took {}m{}s.".format(total // 60, total % 60))
    print("cv mean {}: {:.4f}".format(metric_str, mean_loss))

    print("Bootstrapping...B)")
    bs_losses = []
    for i in range(10):
        print("Sample {}".format(i))
        data = np.hstack((X, Y.reshape(len(Y), 1)))
        train = resample(data, n_samples=int(0.7*len(X)))
        test = np.array([sample for sample in data if sample.tolist() not in train.tolist()])
        X_train, Y_train = train[:,:-1], train[:,-1]
        X_test, Y_test = test[:,:-1], test[:,-1]
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        bs_losses.append(metric(Y_test, Y_pred))
    mean_loss = np.mean(bs_losses)
    n = len(bs_losses)
    print("bootstrap mean {}: {:.4f}".format(metric_str, mean_loss))
    lower = mean_loss - 1.96*np.std(bs_losses)/np.sqrt(n)
    upper = mean_loss + 1.96*np.std(bs_losses)/np.sqrt(n)
    print("95% confidence interval: [{:.4f}, {:.4f}]".format(lower, upper))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sentiment Analysis")
    parser.add_argument('model', help="model")
    parser.add_argument('-i', dest='imp_method', help="imputation method")
    parser.add_argument('-m', dest='fs_method', help="feature selection method")
    parser.add_argument('-o', dest='outcome', help="outcome (i.e. gpa)")
    parser.add_argument('-k', dest='k', help="k", type=int, default='100')
    parser.add_argument('-d', dest='data_dir', help='data directory', default='data')
    args = parser.parse_args()

    features_path = 'features_{}_{}.csv'.format(args.outcome, args.imp_method)
    labels_path = 'labels_{}_{}.csv'.format(args.outcome, args.imp_method)

    print("Loading dataset...")
    X, Y = get_data(args.data_dir, features_path, labels_path)
    is_classf = Y.dtype == np.int8 
    print("Successfully loaded dataset.")
    if args.fs_method:
        print("Performing feature selection using {}...".format(args.fs_method))
        X = feature_selection(X, Y, args.outcome, args.fs_method, args.data_dir, verbose=True)
    print("X dim: {}".format(X.shape))
    train(args.model, X, Y, is_classf)

