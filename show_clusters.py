import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import mplleaflet as mpl
workdir = 'C:\\Users\\nirfi\\Downloads\\taxi.train.csv\\'

df = pd.read_csv(workdir + 'kusomo.csv')
points = zip(df.to_longitude, df.to_latitude) + zip(df.from_longitude, df.from_latitude)
kmeans = KMeans(init='k-means++', n_clusters=20, n_init=15)

partial = points[:100000]
h = 0.002
kmeans.fit(partial)
# x_min, x_max = min((x[0] for x in partial)), max((x[0] for x in partial))
# y_min, y_max = min((y[1] for y in partial)), max((y[1] for y in partial))
x_min, x_max = -74.5, -73.23
y_min, y_max = 40.34, 41.1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.contour(xx, yy, Z, LineWidth = 4, extent=(xx.min(), xx.max(), yy.min(), yy.max()))
# plt.imshow(Z, interpolation='nearest',
#            extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#            cmap=plt.cm.Paired,
#            aspect='auto', origin='lower', alpha=0.5)

mpl.show()