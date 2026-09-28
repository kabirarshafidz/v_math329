import os

import numpy as np
from scipy.io import loadmat

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(CODE_DIR, '..', 'data', 'mnist_train_test.mat')
RESULTS_DIR = os.path.join(CODE_DIR, '..', 'results')


def load_data(path=DATA_PATH, verbose=False):
    data = loadmat(path, squeeze_me=True, struct_as_record=False)
    train, test = data['train'], data['test']

    if verbose:
        print(train._fieldnames)
        for name in train._fieldnames:
            val = getattr(train, name)
            print(name, type(val), getattr(val, 'shape', None), getattr(val, 'dtype', None))

    X = train.X.T
    y = train.y.astype(np.float64)
    return X, y
