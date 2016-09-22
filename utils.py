import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt


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


def calc_clusters(points, init='k-means++', n_clusters=200, n_init=1):
    # points is a tuple of 2 arrays - long and lat
    est = KMeans(init=init, n_clusters=n_clusters, n_init=n_init)
    est.fit(points)

    joblib.dump(est, 'kmeans_estimator.joblib.pkl', compress=9) # save the classifier
    return est.labels_, est.cluster_centers_


def stich_coordinates(train_df, valid_df, test_df):
    long_coords = []
    lat_coords = []
    for df in [train_df, valid_df, test_df]:
        long_coords += df.from_longitude.tolist() + df.to_longitude.tolist()
        lat_coords += df.from_latitude.tolist() + df.to_latitude.tolist()

    return long_coords, lat_coords


def add_labels_to_data(labels, train_df, valid_df, test_df):
    x = len(train_df)
    y = len(valid_df)
    z = len(test_df)

    train_df.loc[:, 'c_in'] = labels[:x]
    train_df.loc[:, 'c_out'] = labels[x:2*x]
    valid_df.loc[:, 'c_in'] = labels[2*x:2*x + y]
    valid_df.loc[:, 'c_out'] = labels[2*x + y:2*x + 2*y]
    test_df.loc[:, 'c_in'] = labels[2*x + 2*y:2*x + 2*y + z]
    test_df.loc[:, 'c_out'] = labels[2*x + 2*y + z:2*x + 2*y + 2*z]

    train_df.to_pickle(os.path.join(parent_dir, 'taxi.train.nyc.pkl'))
    valid_df.to_pickle(os.path.join(parent_dir, 'taxi.valid.pkl'))
    test_df.to_pickle(os.path.join(parent_dir, 'taxi.test.no.label.pkl'))


def train_regressors(train_df):
    # for
    pass


def draw_clusters_heatmap(df):
    mat = np.zeros((200, 200))
    for index, row in df.iterrows():
        mat[int(row.c_in), int(row.c_out)] += 1
    np.savetxt('mat.csv', mat, delimiter=',')
def get_day_class(date_str):
    weekday = pd.Timestamp(date_str).isoweekday()
    if weekday in [2, 3, 4]:
        # Tuesday - Thursday
        return 0
    elif weekday in [5, 6, 7]:
        # Friday - Sunday -> 1-3
        return weekday - 4
    else:
        # Monday
        return 4

    
