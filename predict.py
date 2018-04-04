import argparse
import h5py
import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression

from feature_selection import feature_selection
from utils import get_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fragile Families Predict Script")
    parser.add_argument('model', help="model")
    parser.add_argument('-i', dest='imp_method', help="imputation method", default='Mode')
    parser.add_argument('-m', dest='fs_method', help="feature selection method", default='ElasticNet')
    parser.add_argument('-d', dest='data_dir', help='data directory', default='data')
    parser.add_argument('-s', dest='results_dir', help='results directory', default='results')
    args = parser.parse_args()

    outcomes = ['gpa', 'grit', 'materialHardship', 'eviction', 'layoff', 'jobTraining']
    predictions = [np.arange(1, 4243)]
    for outcome in outcomes:
        print("Generating predictions for {}...".format(outcome))
        train_features_path = 'features_{}_{}.csv'.format(outcome, args.imp_method)
        selected_row_ids_path = 'ids_{}_{}.csv'.format(outcome, args.imp_method)
        selected_features_path = 'feature_subset_{}_{}_{}.h5'.format(outcome, args.fs_method, args.imp_method)
        test_features_path = 'test_features_{}_{}.csv'.format(outcome, args.imp_method)
        X_test = pd.read_csv(os.path.join(args.data_dir, test_features_path), low_memory=False).as_matrix()
        labels_path = 'labels_{}_{}.csv'.format(outcome, args.imp_method)
        with h5py.File(os.path.join(args.data_dir, selected_features_path), 'r') as hf:
            selected_features = hf[args.fs_method][:]
        ids = np.genfromtxt(os.path.join(args.data_dir, selected_row_ids_path), delimiter=',', skip_header=1)
        ids = ids.astype(np.int8)
        print("Loading datasets...")
        X_train, Y = get_data(args.data_dir, train_features_path, labels_path)
        X_train = X_train[:,selected_features] 
        X_test = X_test[:,selected_features]
        X = np.vstack([X_train, X_test])
        is_classf = Y.dtype == np.int8 
        print("Successfully loaded datasets.")
        print("Fitting model...")
        model = LogisticRegression() if is_classf else LinearRegression()
        model.fit(X_train, Y)
        Y_pred = model.predict_proba(X_test)[:,1] if is_classf else model.predict(X_test)
        y_mean = np.mean(Y_pred)
        pred_outcome = np.repeat(y_mean, 4242)
        j = 0
        for i in ids:
            pred_outcome[i] = Y_pred[j]
            j += 1
        predictions.append(pred_outcome)
    
    predictions = np.array(predictions).T
    pred_df = pd.DataFrame(data=predictions, columns=['challengeID']+outcomes)
    pred_df['challengeID'] = pred_df['challengeID'].astype('int64')
    pred_df.to_csv(os.path.join(args.results_dir, 'prediction.csv'), index=False)

