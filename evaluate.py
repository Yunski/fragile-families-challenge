import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

model_id = {'Linear':0, 'Ridge':1, 'AdaBoost':2, 'RandomForest':3, 'SVM':4}
outcome_id = {'gpa':0,'grit':1,'materialHardship':2, 'eviction':3, 'layoff':4, 'jobTraining':5}
fs_id = {'None':0, 'ElasticNet':5, 'RFE':10, 'PCA':15}
def display(results_dir):
    X_knn = [[0]*6 for i in range(20)]
    X_mode = [[0]*6 for i in range(20)]
    for f in sorted(glob.glob(os.path.join(results_dir, '*.json'))):
        fields = f.split('_')[1].split('-')
        model, outcome, fs_method = fields[0], fields[1], fields[2]
        imp_method = fields[3].split('.')[0]
        X = X_knn if imp_method == 'KNN' else X_mode 
        m = fs_id[fs_method] + model_id[model]
        o = outcome_id[outcome]
        metrics = json.load(open(f))
        X[m][o] = "{:.4f}".format(metrics['bootstrap_brier_loss']) if 'bootstrap_brier_loss' in metrics.keys() \
        else "{:.4f}".format(metrics['bootstrap_mse'])

    for x in X_knn:
        print('& ' + '& '.join([i+' ' for i in x]))
    print()
    for x in X_mode:
        print('& ' + '& '.join([i+' ' for i in x]))

    """
    X_knn = np.array(X_knn)
    X_mode = np.array(X_mode)
    knn_df = pd.DataFrame(data=X_knn)
    mode_df = pd.DataFrame(data=X_mode)
    knn_df.to_csv(os.path.join(results_dir, 'results_knn.csv'), index=False)
    mode_df.to_csv(os.path.join(results_dir, 'results_mode.csv'), index=False)
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fragile Families Results")
    parser.add_argument('-s', dest='results_dir', help='results directory', default='results')
    args = parser.parse_args()
    display(args.results_dir)

