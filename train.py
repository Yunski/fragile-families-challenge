import argparse
import json
import numpy as np
import pandas as pd
import os
import sys
import scipy
import time

from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import RidgeCV, RidgeClassifierCV, Ridge, RidgeClassifier, LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.svm import SVR, SVC 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import resample

from feature_selection import feature_selection
from utils import get_data

def train(classifier, X, Y, is_classf, outcome, fs_method, imp_method, 
    data_dir, results_dir, cv=10, verbose=0):
    results_path = os.path.join(results_dir, 
        'score_{}-{}-{}-{}.json'.format(classifier, outcome, fs_method, imp_method))
    if os.path.exists(results_path):
        if verbose:
            print("Model already trained. See {}".format(results_path))
        return
    if classifier == 'Linear':
        if is_classf:
            model = LogisticRegression()
        else:
            model = LinearRegression()
    elif classifier == 'Ridge':
        if is_classf:
            model = RidgeClassifierCV(alphas=(1e-3, 1e-2, 1e-1, 1, 10, 100))
        else:
            model = RidgeCV(alphas=(1e-3, 1e-2, 1e-1, 1, 10, 100))
        print("Finding best alpha...")
        model.fit(X, Y)
        best_alpha = model.alpha_
        print("Best alpha: {}".format(model.alpha_))
        if is_classf:
            model = RidgeClassifier(alpha=best_alpha) 
        else:
            model = Ridge(alpha=best_alpha) 
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
    elif classifier == 'SVM':
        if is_classf:
            model = SVC(kernel='linear', probability=True)
        else:
            model = SVR(kernel='linear')
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
        if classifier == 'Ridge' and is_classf:
            d = model.decision_function(X_test)
            Y_pred = np.exp(d) / (1 + np.exp(d))
        else:
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
        if classifier == 'Ridge' and is_classf:
            # from https://stackoverflow.com/questions/22538080/scikit-learn-ridge-classifier-extracting-class-probabilities
            d = model.decision_function(X_test)
            Y_pred = np.exp(d) / (1 + np.exp(d))
        else:
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

    with open(results_path, 'w') as f: 
        json.dump(scores, f)
        if verbose:
            print("Successfully saved scores.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fragile Families Train Script")
    parser.add_argument('model', help="model")
    parser.add_argument('outcome', help="outcome")
    parser.add_argument('-i', dest='imp_method', help="imputation method", default='KNN')
    parser.add_argument('-m', dest='fs_method', help="feature selection method", default='ElasticNet')
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
    
