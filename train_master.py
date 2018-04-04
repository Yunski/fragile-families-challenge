import argparse
import numpy as np

from feature_selection import feature_selection
from train import train
from utils import get_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fragile Families Pipeline")
    parser.add_argument('-d', dest='data_dir', help="data directory", default='data')
    parser.add_argument('-s', dest='results_dir', help='results directory', default='results')

    args = parser.parse_args()

    fs_methods = ['PCA', 'RFE']#, 'ElasticNet', None]
    outcomes = ['gpa', 'grit', 'materialHardship', 'eviction', 'layoff', 'jobTraining']
    models = ['Linear', 'Ridge', 'AdaBoost', 'RandomForest', 'SVM']

    for imp_method in ['Mode', 'KNN']:
        for outcome in outcomes:    
            print("Generating results for {}".format(outcome))
            features_path = 'features_{}_{}.csv'.format(outcome, imp_method)
            labels_path = 'labels_{}_{}.csv'.format(outcome, imp_method)

            print("Loading dataset...")
            X, Y = get_data(args.data_dir, features_path, labels_path)
            is_classf = Y.dtype == np.int8 
            print("Successfully loaded dataset.")
            
            for fs_method in fs_methods:
                if fs_method: 
                    print("Performing feature selection using {}...".format(fs_method))
                    print(X.shape)
                    X_subset = feature_selection(X, Y, outcome, fs_method, imp_method, args.data_dir, verbose=1)
                    print(X_subset.shape)
                """
                for model in models:
                    train(model, X_subset, Y, is_classf, outcome, fs_method, 
                        imp_method, args.data_dir, args.results_dir, verbose=1)
                """