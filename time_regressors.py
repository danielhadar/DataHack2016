import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.externals import joblib

workdir = 'C:\\Users\\nirfi\\Downloads\\taxi.train.csv\\'
# df = pd.read_csv(workdir + 'taxi.train.csv')
# df['time'] = df.from_datetime.apply(lambda x: pd.Timestamp(x).hour * 3600 + pd.Timestamp(x).minute * 60 + pd.Timestamp(x).second)
# kmeans = KMeans(init='k-means++', n_clusters=6, n_init=15)
# kmeans.fit(np.array(df.time)[np.newaxis].T)
# valid = pd.read_csv(workdir + 'taxi.valid.csv')
# valid['time'] = valid.from_datetime.apply(lambda x: pd.Timestamp(x).hour * 3600 + pd.Timestamp(x).minute * 60 + pd.Timestamp(x).second)
# joblib.dump(kmeans, workdir+'kmeans.joblib.pkl')
# joblib.dump(df, workdir+'train.joblib.pkl')
# joblib.dump(valid, workdir+'valid.joblib.pkl')

kmeans = joblib.load(workdir+'kmeans.joblib.pkl')
df = joblib.load(workdir+'train.joblib.pkl')
valid = joblib.load(workdir+'valid.joblib.pkl')

df['clsts'] = kmeans.predict(np.array(df.time).reshape(-1, 1))
df = df.drop(['RatecodeFactor', 'PaymentFactor', 'from_datetime'], axis=1)

models = {}
# models = joblib.load(workdir+'regressors.joblib.pkl')
for i in range(6):
    models[i] = GradientBoostingRegressor(n_estimators=100,
                                           learning_rate=0.1,
                                           subsample=.5,
                                           max_depth=2)
    models[i].fit(df.drop('y', axis=1).loc[df.clsts == i], df.y.loc[df.clsts == i])
# valid = pd.read_csv(workdir + 'taxi.valid.csv')
# valid['time'] = valid.from_datetime.apply(lambda x: pd.Timestamp(x).hour * 3600 + pd.Timestamp(x).minute * 60 + pd.Timestamp(x).second)
# valid['clsts'] = kmeans.predict(np.array(valid.time)[np.newaxis].T)
# valid = valid.drop([['RatecodeFactor', 'PaymentFactor', 'from_datetime']], axis=1)
# valid['pred_y'] = valid.drop('y', axis=1).apply(lambda x: models[x.clsts].staged_predict())

# joblib.dump(models, workdir+'regressors.joblib.pkl')
valid['clsts'] = kmeans.predict(np.array(valid.time).reshape(-1, 1 ))
valid = valid.drop(['RatecodeFactor', 'PaymentFactor', 'from_datetime'], axis=1)
# valid['pred_y'] = valid.drop('y', axis=1).apply(lambda x: models[x.clsts].staged_predict(x.drop('clsts', axis=1)))
clsts = valid.clsts.copy()
# valid = valid.drop('clsts',axis=1)
res = pd.DataFrame(columns = ['y', 'pred_y'])
for i in range(6):
    cur = valid.loc[valid.clsts == i].copy()
    cur_res = cur.y.copy().to_frame()
    cur = cur.drop('y', axis=1)
    cur_res['pred_y'] = models[i].predict(cur)
    cur_res.to_csv(workdir + 'no' + str(i) + 'res.csv')
    res.append(cur_res)
# pred_y = [models[row.clsts].staged_predict(row.drop('clsts', axis=1)) for row in valid.iterrows()]

# res['pred_y'] = [models[clsts.loc[i]].predict(valid.loc[i]) for i in range(len(clsts))]
res.to_csv(workdir + 'res.csv')