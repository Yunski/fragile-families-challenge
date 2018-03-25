import argparse
import numpy as np

from feature_selection import feature_selection
from train import train
from utils import get_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sentiment Analysis")
    parser.add_argument('imp_method', help="imputation method")
    parser.add_argument('-d', dest='data_dir', help="data directory", default='data')
    parser.add_argument('-s', dest='results_dir', help='results directory', default='results')

    args = parser.parse_args()

    fs_methods = [None, 'PCA', 'ElasticNet', 'Lasso', 'RFE', 'ANOVA']
    outcomes = ['gpa', 'grit', 'materialHardship', 'eviction', 'layoff', 'jobTraining']
    models = ['Lasso', 'AdaBoost', 'RandomForest', 'GP', 'SVM']
    fs_models = models.copy()
    fs_models.remove('Lasso') 

    for outcome in outcomes:    
        features_path = 'features_{}_{}.csv'.format(outcome, args.imp_method)
        labels_path = 'labels_{}_{}.csv'.format(outcome, args.imp_method)

        print("Loading dataset...")
        X, Y = get_data(args.data_dir, features_path, labels_path)
        is_classf = Y.dtype == np.int8 
        print("Successfully loaded dataset.")
        
        for fs_method in fs_methods:
            if fs_method:
                print("Performing feature selection using {}...".format(fs_method))
                use_full_data = False
                X = feature_selection(X, Y, outcome, fs_method, args.imp_method, args.data_dir, verbose=1)
            else:
                use_full_data = True
            print("X dim: {}".format(X.shape))

            models = models if use_full_data else fs_models
            for model in models:
                train(model, X, Y, is_classf, outcome, fs_method, 
                    args.imp_method, args.data_dir, args.results_dir, verbose=1)
   