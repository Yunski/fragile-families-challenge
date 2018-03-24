import os
import numpy as np
import pandas as pd

def get_data(data_dir, features_filename, labels_filename):
    X = pd.read_csv(os.path.join(data_dir, features_filename), low_memory=False).as_matrix()
    Y = pd.read_csv(os.path.join(data_dir, labels_filename))
    if Y.iloc[:,0].nunique() == 2:
       Y.iloc[:,0] = Y.iloc[:,0].astype('int8')
    Y = np.squeeze(Y.as_matrix())
    return X, Y
