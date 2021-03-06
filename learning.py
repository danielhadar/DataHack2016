from sklearn.cross_validation import ShuffleSplit
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action = "ignore", category = FutureWarning)

n_est = 1000


def final_run(data, len_train, len_valid, len_test):
    geo_y, geo_models = predict_geo_y(data, [i for i in range(len_train)], [j for j in range(len_train, len_train + len_valid)])
    temporal_y, temporal_models = predict_temporal_y(data, [i for i in range(len_train)], [j for j in range(len_train, len_train + len_valid)])

    import pickle
    pickle.dump(geo_y, open('geo_y.pkl', 'wb'))
    pickle.dump(temporal_y, open('temporal_y.pkl', 'wb'))
    pickle.dump(geo_models, open('geo_model.pkl', 'wb'))
    pickle.dump(temporal_models, open('temporal_model.pkl', 'wb'))
    # geo_y = pickle.load(open('geo_y.pkl', 'rb'))
    # temporal_y = pickle.load(open('temporal_y.pkl', 'rb'))
    # geo_models = pickle.load(open('geo_model.pkl', 'rb'))
    # temporal_models = pickle.load(open('temporal_model.pkl', 'rb'))

    print("starting 2nd learner")
    regr = second_learner(np.array([geo_y, temporal_y]).T, data.iloc[len_train:len_train + len_valid].y)

    print("working on test data")
    test_geo_y = []
    test_temporal_y = []
    test_data = data.iloc[len_train + len_valid:]
    for idx, (_, test_sample) in enumerate(test_data.iterrows()):
        if idx % 10000 == 0:
            print(idx)
        i = test_sample.c_in
        j = test_sample.c_out
        while not geo_models[test_sample.c_in][test_sample.c_out]:
            i += 1
            j += 1
            geo_models[test_sample.c_in][test_sample.c_out] = geo_models[i % 10][j % 10]
        test_geo_y.append(geo_models[test_sample.c_in][test_sample.c_out].predict(
            test_sample.drop(['Unnamed: 0', 'VendorID', 'from_datetime',
                              'passenger_count', 'RatecodeFactor', 'PaymentFactor',
                              'y', 'c_in', 'c_out', 'c_time']))[0])
        test_temporal_y.append(temporal_models[test_sample.c_time].predict(
            test_sample.drop(['Unnamed: 0', 'VendorID', 'from_datetime',
                              'passenger_count', 'RatecodeFactor', 'PaymentFactor',
                              'y', 'c_in', 'c_out', 'c_time']))[0])

    print("starting 2nd learner")
    final_y = regr.predict(np.array([test_geo_y, test_temporal_y]).T)

    np.savetxt("final_y.csv", final_y, delimiter=",")


def cross_validation(data, validation_percent):
    final_results = []
    rs = ShuffleSplit(len(data), n_iter=10, test_size=int(len(data) * validation_percent))
    for train_index, test_index in rs:
        test_data = data.iloc[test_index]
        geo_y = predict_geo_y(data, train_index, test_index)
        temporal_y = predict_temporal_y(data, train_index, test_index)

        predicted_y = second_learner(np.array([geo_y, temporal_y]).T, test_data.y)

        currrrrr_results = np.mean(np.power(np.array(predicted_y)-np.array(test_data.y),2))
        final_results.append(currrrrr_results)
        print(
            np.mean(np.power(np.array(geo_y) - np.array(test_data.y), 2)),
            np.mean(np.power(np.array(temporal_y) - np.array(test_data.y), 2))
        )
        print(currrrrr_results)
    print(final_results)


def predict_geo_y(data, train_index, test_index):
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]
    models = [[[] for _ in range(10)] for _ in range(10)]
    predicted_y_list = []
    for i in range(10):
        for j in range(10):
            cur_data = train_data.loc[(train_data.c_in == i) & (train_data.c_out == j)]
            if not cur_data.empty:
                print(i, j, len(cur_data))
                models[i][j] = GradientBoostingRegressor(n_estimators=n_est, learning_rate=0.1, subsample=.5,
                                                         max_depth=2). \
                    fit(cur_data.drop(['Unnamed: 0', 'VendorID', 'from_datetime', 'passenger_count', 'RatecodeFactor',
                                       'PaymentFactor', 'y', 'c_in', 'c_out', 'c_time'], axis=1), cur_data.y)

    for idx, (_, test_sample) in enumerate(test_data.iterrows()):
        if idx % 10000 == 0:
            print(idx)
        i = test_sample.c_in
        j = test_sample.c_out
        while not models[test_sample.c_in][test_sample.c_out]:
            i += 1
            j += 1
            models[test_sample.c_in][test_sample.c_out] = models[i % 10][j % 10]
        predicted_y_list.append(models[test_sample.c_in][test_sample.c_out].predict(
            test_sample.drop(['Unnamed: 0', 'VendorID', 'from_datetime',
                              'passenger_count', 'RatecodeFactor', 'PaymentFactor',
                              'y', 'c_in', 'c_out', 'c_time']))[0])

    return predicted_y_list, models


def predict_temporal_y(data, train_index, test_index, k=6):
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]
    models = {}
    predicted_y_list = []

    for i in range(k):
        print(i)
        models[i] = GradientBoostingRegressor(n_estimators=n_est, learning_rate=0.1, subsample=.5, max_depth=2)
        models[i].fit(train_data.drop(['Unnamed: 0', 'VendorID', 'from_datetime', 'passenger_count', 'RatecodeFactor',
                                       'PaymentFactor', 'y', 'c_in', 'c_out', 'c_time'], axis=1)
                      .loc[train_data.c_time == i], train_data.y.loc[train_data.c_time == i])

    for idx, (_, test_sample) in enumerate(test_data.iterrows()):
        if idx % 10000 == 0:
            print(idx)
        predicted_y_list.append(models[test_sample.c_time].predict(
            test_sample.drop(['Unnamed: 0', 'VendorID', 'from_datetime',
                              'passenger_count', 'RatecodeFactor', 'PaymentFactor',
                              'y', 'c_in', 'c_out', 'c_time']))[0])

    return predicted_y_list, models


def second_learner(feat, y):
    regr = LinearRegression()
    print(np.shape(feat))
    print(np.shape(y))
    np.savetxt("feat.csv", feat, delimiter=",")
    np.savetxt("y.csv", y, delimiter=",")
    regr.fit(feat, y)
    print("done fit 2nd learner")
    return regr

