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
    # print("make pickle")
    # utils.make_pickles()
    # train_df, valid_df, test_df = utils.loadings('pkl')
    # long_coords, lat_coords = utils.stich_coordinates(train_df, valid_df, test_df)
    # print("calc clusters")
    # geo_labels, centers, est = utils.calc_clusters(np.column_stack((long_coords, lat_coords)), n_clusters=n_clusters, n_init=10)
    # time_labels = utils.add_c_time_column(train_df.time.tolist() + valid_df.time.tolist() + test_df.time.tolist())
    # print("add lables")
    # utils.add_labels_to_data(geo_labels, time_labels, train_df, valid_df, test_df)
    # print("draw heatmap")
    # utils.draw_clusters_heatmap(train_df, n_clusters=n_clusters)


    train_df, valid_df, test_df = utils.loadings('pkl')
    # learning.cross_validation(pd.concat([train_df, valid_df]), validation_percent=.033)
    learning.final_run(pd.concat([train_df, valid_df, test_df]), len(train_df)+len(valid_df), len(test_df))








quit()


# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = est.predict(np.c_[xx.ravel(), yy.ravel()])


# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(data[:, 0], data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = est.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()