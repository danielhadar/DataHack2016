import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
import utils
import learning
import pandas as pd
from show_clusters import show_clusters

n_clusters = 10

if __name__ == '__main__':

    # >> run only once (to add permanent cluster labels to the data) <<
    print("make pickle")
    utils.make_pickles()
    train_df, valid_df, test_df = utils.loadings('pkl')
    long_coords, lat_coords = utils.stich_coordinates(train_df, valid_df, test_df)
    print("calc clusters")
    geo_labels, centers, est = utils.calc_clusters(np.column_stack((long_coords, lat_coords)), n_clusters=n_clusters, n_init=10)
    time_labels = utils.add_c_time_column(train_df.time.tolist() + valid_df.time.tolist() + test_df.time.tolist())
    print("add lables")
    utils.add_labels_to_data(geo_labels, time_labels, train_df, valid_df, test_df)
    print("draw heatmap")
    utils.draw_clusters_heatmap(train_df, n_clusters=n_clusters)

    train_df, valid_df, test_df = utils.loadings('pkl')
    # learning.cross_validation(pd.concat([train_df, valid_df]), validation_percent=.033)
    learning.final_run(pd.concat([train_df, valid_df, test_df]), len(train_df), len(valid_df), len(test_df))