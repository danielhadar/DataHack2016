from sklearn.cross_validation import ShuffleSplit
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action = "ignore", category = FutureWarning)


def cross_validation(data, validation_percent):
    final_results = []
    rs = ShuffleSplit(len(data), n_iter=10, test_size=int(len(data) * validation_percent))
    for train_index, test_index in rs:
        test_data = data.iloc[test_index]
        geo_y = predict_geo_y(data, train_index, test_index)
        temporal_y = predict_temporal_y(data, train_index, test_index)

        predicted_y = second_learner([geo_y, temporal_y], test_data.y)


        final_results.append(np.mean(np.power(np.array(predicted_y)-np.array(test_data.y),2)))
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
                models[i][j] = GradientBoostingRegressor(n_estimators=500, learning_rate=0.1, subsample=.5,
                                                         max_depth=2). \
                    fit(cur_data.drop(['Unnamed: 0', 'VendorID', 'from_datetime', 'passenger_count', 'RatecodeFactor',
                                       'PaymentFactor', 'y', 'c_in', 'c_out'], axis=1), cur_data.y)

    for idx, (_, test_sample) in enumerate(test_data.iterrows()):
        predicted_y_list.append(models[test_sample.c_in][test_sample.c_out].predict(
            test_sample.drop(['Unnamed: 0', 'VendorID', 'from_datetime',
                              'passenger_count', 'RatecodeFactor', 'PaymentFactor',
                              'y', 'c_in', 'c_out']))[0])

    return predicted_y_list


def predict_temporal_y():
    pass


def second_learner(feat, y):
    pass