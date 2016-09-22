import os
import pandas as pd

# constants
parent_dir = '.'


def loadings(type):
    # type is 'csv' or 'pkl'
    if type == 'csv':
        train_df = pd.read_csv(os.path.join(parent_dir, 'taxi.train.nyc.csv'))
        valid_df = pd.read_csv(os.path.join(parent_dir, 'taxi.valid.csv'))
        test_df = pd.read_csv(os.path.join(parent_dir, 'taxi.test.no.label.csv'))
    elif type == 'pkl':
        train_df = pd.read_pickle(os.path.join(parent_dir, 'taxi.train.nyc.pkl'))
        valid_df = pd.read_pickle(os.path.join(parent_dir, 'taxi.valid.pkl'))
        test_df = pd.read_pickle(os.path.join(parent_dir, 'taxi.test.no.label.pkl'))

    return train_df, valid_df, test_df
