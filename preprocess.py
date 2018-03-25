import argparse
import h5py
import os
import numpy as np
import pandas as pd

from fancyimpute import KNN
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import Imputer, StandardScaler
from tqdm import tqdm

def preprocess(method, data_dir, force):
    if method not in ['Mode', 'KNN']:
        print("{} not supported.".format(method))
        return

    if os.path.exists(os.path.join(data_dir, 'data_{}.h5'.format(method))) and \
        os.path.exists(os.path.join(data_dir, 'data_columns_{}'.format(method))) and \
        not force:
        print("Data file already exists for {}. Loading...".format(method))
        with h5py.File(os.path.join(data_dir, 'data_{}.h5'.format(method)), 'r') as hf:
            X = hf['data_{}'.format(method)][:]
        with open(os.path.join(data_dir, 'data_columns_{}'.format(method)), 'r') as f:
            columns = f.read().split(',')
    else:
        print("Reading data...")
        df = pd.read_csv(os.path.join(data_dir, 'background.csv'), low_memory=False)
        print("complete")

        f =  open(os.path.join(data_dir, 'constantVariables.txt'), 'r')
        constants = set([line.strip('\n') for line in f] + ['challengeID'])

        print("Remove constants.")
        constants_idx = []
        for i, label in enumerate(df.columns):
            if (label in constants):
                constants_idx.append(i)

        df.drop(df.columns[constants_idx], axis=1, inplace=True)
        print(df.shape)

        print("Remove non-numeric.")
        data_type = ['float64', 'int64']
        df = df.select_dtypes(include = data_type)
        print(df.shape)

        o_dtypes = df.dtypes.copy()

        print("Remove NA.")
        df.dropna(axis=1, how='any', inplace=True)
        print(df.shape)

        df[df < 0] = np.nan
        df.dropna(axis=1, how='all', inplace=True)
        print(df.shape)    

        X = df.as_matrix().astype(np.float32)
        print("{} Imputation...".format(method))
        if method == 'Mode':
            imp = Imputer(strategy='most_frequent')
            X = imp.fit_transform(X)
        elif method == 'KNN':    
            X = KNN(k=3).complete(X)
        else:
            raise Exception("Invalid method found.")
        print("Finished.")
     
        print("Remove constants.")
        selector = VarianceThreshold()
        X = selector.fit_transform(X)
        print(X.shape)         

        df_impute = pd.DataFrame(X, columns=df.columns[selector.get_support()])

        print("Convert ints to categorical variables.")
        category_idx = [o_dtypes[col].name == 'int64' for col in df_impute.columns]
        category_idx = np.array(category_idx and df_impute.apply(lambda x: x.nunique() <= 5, axis=0).tolist())
        categories = df_impute.iloc[:,category_idx]
        categories = categories.apply(lambda col: col.astype('category'))
        ind_columns = pd.get_dummies(categories)
        print("Normalize.")
        float_columns = df_impute.iloc[:,~category_idx]
        scaler = StandardScaler()
        normalized_columns = pd.DataFrame(scaler.fit_transform(float_columns), columns=float_columns.columns)
        df_impute = pd.concat([ind_columns, normalized_columns], axis=1)
        columns = df_impute.columns
        X = df_impute.as_matrix()

        print("Writing files {} and {}".format('data_{}.h5'.format(method), 'data_columns_{}'.format(method)))
        with h5py.File(os.path.join(data_dir, 'data_{}.h5'.format(method)), 'w') as hf:
            hf.create_dataset('data_{}'.format(method),  data=X)
        with open(os.path.join(data_dir, 'data_columns_{}'.format(method)), 'w') as f:
            f.write(','.join(df_impute.columns) + '\n')
        print("Finished.")

    df_Y = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    challengeIds = df_Y['challengeID'].values - 1
    Y = df_Y.as_matrix()
    for col in range(1, Y.shape[1]):
        label = df_Y.columns[col]
        print("Generating feature and label csvs for {}...".format(label))
        Y_cur = Y[:,col] 
        non_na = np.isfinite(Y_cur)    
        idx = challengeIds[non_na]
        selector = VarianceThreshold()
        X_cur = X[idx]
        X_cur = selector.fit_transform(X_cur)
        features = pd.DataFrame(X_cur, columns=columns[selector.get_support()])
        features.to_csv(os.path.join(data_dir, 'features_{}_{}.csv'.format(label, method)), index=False)
        labels = pd.DataFrame(Y_cur[non_na], columns=[label])
        labels.to_csv(os.path.join(data_dir, 'labels_{}_{}.csv'.format(label, method)), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fragile Families Data Imputation")
    parser.add_argument('method', help="imputation method", default='KNN')
    parser.add_argument('-d', dest='data_dir', help="data directory", default='data')
    parser.add_argument('-f', dest='force', help="force overwrite of data files", action='store_true')
    args = parser.parse_args()
    preprocess(args.method, args.data_dir, args.force) 

