import argparse
import json
import numpy as np
import pandas as pd
import os
import sys
import scipy
import time

from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression
from sklearn.metrics import brier_score_loss, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVR, SVC 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import resample

from feature_selection import feature_selection
from utils import get_data

def train(classifier, X, Y, is_classf, outcome, fs_method, imp_method, 
    data_dir, results_dir, cv=10, verbose=0):
    if classifier == 'Lasso':
        if is_classf:
            model = LogisticRegression(penalty='l1')
        else:
            model = Lasso()
    elif classifier == 'AdaBoost':
        if is_classf:
            estimator = DecisionTreeClassifier(max_depth=1) 
            model = AdaBoostClassifier(estimator, n_estimators=100) 
        else: 
            estimator = DecisionTreeRegressor(max_depth=1)
            model = AdaBoostRegressor(estimator, n_estimators=100)
    elif classifier == 'RandomForest':
        if is_classf:
            model = RandomForestClassifier(n_estimators=50)
        else:
            model = RandomForestRegressor(n_estimators=50)
    elif classifier == 'GP':
        if is_classf:
            model = GaussianProcessClassifier(normalize_y=True, n_restarts_optimizer=9)
        else:
            model = GaussianProcessRegressor(normalize_y=True, n_restarts_optimizer=9) 
    elif classifier == 'SVM':
        if is_classf:
            model = SVC(kernel='linear', probability=True)
        else:
            model = LinearSVR()
    else:
        raise ValueError("model {} not available".format(classifier))

    scores = {}
    metric = brier_score_loss if is_classf else mean_squared_error
    metric_str = 'brier_loss' if is_classf else 'mse'
    if verbose:
        print("Training {}...".format(classifier))
        print("10-Fold cross validation...")
    start = time.time()

    kf = KFold(n_splits=cv)
    kf.get_n_splits(X)
    losses = []
    for i, (train_index, test_index) in enumerate(kf.split(X)):
        if verbose:
            sys.stdout.write("\rFold {}/{}".format(i+1, cv))
        X_train, Y_train = X[train_index], Y[train_index]
        X_test, Y_test = X[test_index], Y[test_index] 
        model.fit(X_train, Y_train)
        Y_pred = model.predict_proba(X_test)[:,1] if is_classf else model.predict(X_test)
        losses.append(metric(Y_test, Y_pred))
    mean_loss = np.mean(losses)
    scores['cv_{}'.format(metric_str)] = mean_loss

    total = int(time.time()-start)
    if verbose:
        print("\nTraining took {}m{}s.".format(total // 60, total % 60))
        print("cv mean {}: {:.4f}".format(metric_str, mean_loss))
        print("Bootstrapping...B)")
    bs_losses = []
    for i in range(cv):
        if verbose:
            sys.stdout.write("\rSample {}/{}".format(i+1, cv))
        data = np.hstack((np.arange(len(X)).reshape(len(X), 1), X, Y.reshape(len(Y), 1)))
        train = resample(data, n_samples=int(0.7*len(X)))
        train_ids = set(train[:,0].astype(np.int64))
        train = train[:,1:]
        test = np.array([sample[1:] for sample in data if sample[0] not in train_ids])
        X_train, Y_train = train[:,:-1], train[:,-1]
        X_test, Y_test = test[:,:-1], test[:,-1]
        model.fit(X_train, Y_train)
        Y_pred = model.predict_proba(X_test)[:,1] if is_classf else model.predict(X_test)
        bs_losses.append(metric(Y_test, Y_pred))
    mean_loss = np.mean(bs_losses)
    n = len(bs_losses)
    lower = mean_loss - 1.96*np.std(bs_losses)/np.sqrt(n)
    upper = mean_loss + 1.96*np.std(bs_losses)/np.sqrt(n)
    scores['bootstrap_{}'.format(metric_str)] = mean_loss
    scores['bootstrap_95_lower'] = lower
    scores['bootstrap_95_upper'] = upper

    if verbose:
        print("\nbootstrap mean {}: {:.4f}".format(metric_str, mean_loss))
        print("95% confidence interval: [{:.4f}, {:.4f}]".format(lower, upper))

    with open(os.path.join(results_dir, 'score_{}-{}-{}-{}.json'.format(classifier, outcome, 
        fs_method, imp_method)), 'w') as f: 
        json.dump(scores, f)
        if verbose:
            print("Successfully saved scores.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sentiment Analysis")
    parser.add_argument('model', help="model")
    parser.add_argument('-i', dest='imp_method', help="imputation method")
    parser.add_argument('-m', dest='fs_method', help="feature selection method")
    parser.add_argument('-o', dest='outcome', help="outcome (i.e. gpa)")
    parser.add_argument('-k', dest='k', help="k", type=int, default='100')
    parser.add_argument('-d', dest='data_dir', help='data directory', default='data')
    parser.add_argument('-s', dest='results_dir', help='results directory', default='results')
    args = parser.parse_args()

    features_path = 'features_{}_{}.csv'.format(args.outcome, args.imp_method)
    labels_path = 'labels_{}_{}.csv'.format(args.outcome, args.imp_method)

    print("Loading dataset...")
    X, Y = get_data(args.data_dir, features_path, labels_path)
    is_classf = Y.dtype == np.int8 
    print("Successfully loaded dataset.")
    if args.fs_method:
        print("Performing feature selection using {}...".format(args.fs_method))
        X = feature_selection(X, Y, args.outcome, args.fs_method, args.imp_method, args.data_dir, verbose=1)
    print("X dim: {}".format(X.shape))
    train(args.model, X, Y, is_classf, args.outcome, args.fs_method, 
        args.imp_method, args.data_dir, args.results_dir, verbose=1)    
    
