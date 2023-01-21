import numpy as np
from keras.models import load_model
import pandas as pd
from get_stock_data import *
model_path = "models/trained_data2_acc_0.556372344493866.h5"


def test_1():
    testing_data = np.genfromtxt('testing_data.csv', delimiter=',')
    print(testing_data.shape)

    model = load_model(model_path)
    features, labels = get_features_and_labels(testing_data)
    print(features.shape)

    preds = model.predict(features)
    preds[preds>=0.5] = 1
    preds[preds<0.5] = 0
    df = pd.DataFrame({"prediction": preds.flatten(), "truth": labels.flatten()})
    df.to_csv("preds_truths.csv")
    counter = 0
    for i in range(preds.shape[0]):
        if preds[i][0] == labels[i][0]:
            counter += 1

    print("Accuracy: {}".format((counter/labels.shape[0]) * 100))


def test_2():
    test_ticker = "TSLA"
    path = os.path.join(os.pardir, "testing_data/")
    print(path)
    downlaod_stock_histories(path, [test_ticker])
    path2 = path + "individual_stock_data/"
    print(path2)
    model = load_model(model_path)
    testing_data = get_one_stocks_data(test_ticker, path=path2)
    df = pd.DataFrame(testing_data, columns=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17'])
    df.dropna()
    testing_data = np.array(df)
    testing_data = testing_data[20:]
    features, labels = get_features_and_labels(testing_data)

    features = features[:-1]
    labels = labels[:-1]
    labels[labels == 'down'] = 0
    labels[labels == 'up'] = 1

    features = features.astype(np.float32)
    labels = labels.astype(int)

    preds = model.predict(features)
    results = filter_predictions(preds, labels)
    # preds[preds<0.5] = 0
    # preds[preds >= 0.5] = 1
    # results = np.column_stack((labels, preds))

    # print(preds)
    # print(preds.shape)

    # # df = pd.DataFrame()
    # custom_pred = preds[~mask]
    # print(custom_pred.shape)
    # custom_pred[custom_pred[1] >= 0.8] = 1
    # custom_pred[custom_pred[1] <= 0.2] = 0
    # print(custom_pred)

    df = pd.DataFrame({"prediction": results[:, -1], "truth": results[:, -2]})
    print(df)
    df.to_csv("preds_truths.csv")
    counter = 0
    for i in range(results.shape[0]):
        if results[i][0] == results[i][1]:
            counter += 1

    print("Accuracy: {}".format((counter/results.shape[0]) * 100))


def filter_predictions(preds, labels, l_thresh = 0.2, h_thresh = 0.8):
    data = np.column_stack((labels, preds))
    indexes = []
    for i in range(data.shape[0]):
        if data[i][1] >= h_thresh:
            data[i][1] = 1
            indexes.append(i)
        elif data[i][1] <= l_thresh:
            data[i][1] = 0
            indexes.append(i)
    print(data.shape)
    new_data = []
    for ind in indexes:
        new_data.append(data[ind])
    new_data = np.array(new_data)

    print(new_data.shape)
    print(new_data)
    return new_data


def test_3():
    # tickers_list = ["AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "TSLA", "INTC", "CMCSA", "CSCO", "NVDA", "PYPL",
    #                 "ADBE", "AMGN", "TXN", "AVGO", "GILD", "CHTR", "BIDU", "MU"]
    # tickers_list = get_sp500_tickers()[50:100]
    df = pd.read_csv("russel_1000_tickers.csv")
    df = df[['Symbol']]
    tickers_list = np.array(df)
    tickers_list = tickers_list.flatten()
    tickers_list = list(tickers_list)
    path = os.path.join(os.pardir, "testing_data/")
    # downlaod_stock_histories(path, tickers_list, period='1y')

    model = load_model(model_path)
    for tick in tickers_list:

        data = get_one_stocks_data(tick, path=path + "/individual_stock_data/")
        if data.shape[0] == 0:
            continue
        test_record = np.array([data[-2]])
        features, _ = get_features_and_labels(test_record)
        features = features.astype(np.float32)
        pred = model.predict(features, verbose=0)

        if 0.2 < pred[0] < 0.8:
            # print(pred[0])
            continue
        # custom_pred = pred[~mask]
        prob = np.copy(pred[0])
        pred[pred >= 0.8] = 1
        pred[pred <= 0.2] = 0

        # pred[pred >= 0.5] = 1
        # pred[pred < 0.5] = 0
        # print("prediction: {} , truth: {}".format(pred[0], labels[-1]))
        print("{} prev_open:{} prev_close: {} current_open: {} Prediction: {} Original_Probability: {} ".format(tick, features[:, [0]], features[:, [1]], features[:, [15]], pred[0], prob))

    # labels[labels == 'down'] = 0
    # labels[labels == 'up'] = 1

    # labels = labels.astype(int)


    # print(data)
    # print(test_record)
    # print(features)
    # pred = model.predict(features)
    # pred[pred >= 0.5] = 1
    # pred[pred < 0.5] = 0
    # print("prediction: {} , truth: {}".format(pred[0], labels[-1]))
    # print("prediction: {}".format(pred[0]))
    # for tick in tickers_list:
    #     data = get_one_stocks_data(tick)
        # print(data.shape)
        # print(tick)
    # testing_data = get_one_stocks_data()
    # print(testing_data.shape)

def get_features_and_labels(data: np.array):
    features = data[:, range(0, data.shape[1] - 1)]
    labels = data[:, [data.shape[1] - 1]]
    return features, labels

test_3()
