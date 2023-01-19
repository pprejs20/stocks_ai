import numpy as np
from keras.models import load_model
import pandas as pd
from get_stock_data import *



def test_1():
    testing_data = np.genfromtxt('testing_data.csv', delimiter=',')
    print(testing_data.shape)

    model = load_model("models/trained_model_acc_0.7907382845878601.h5")
    features = testing_data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
    labels = testing_data[:, [testing_data.shape[1] - 1]]
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
    test_ticker = "SHOP"
    path = os.path.join(os.pardir, "data/")
    downlaod_stock_histories(path, [test_ticker])

    model = load_model("models/trained_model_acc_0.809393048286438.h5")
    testing_data = get_one_stocks_data(test_ticker)
    print(testing_data.shape)

    features = testing_data[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
    labels = testing_data[:, [testing_data.shape[1] - 1]]
    labels[labels == 'down'] = 0
    labels[labels == 'up'] = 1

    features = features.astype(np.float32)
    labels = labels.astype(int)

    print(features)
    print(labels)
    preds = model.predict(features)
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0
    df = pd.DataFrame({"prediction": preds.flatten(), "truth": labels.flatten()})
    df.to_csv("preds_truths.csv")
    counter = 0
    for i in range(preds.shape[0]):
        if preds[i][0] == labels[i][0]:
            counter += 1

    print("Accuracy: {}".format((counter/labels.shape[0]) * 100))


def test_3():
    tickers_list = ["AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "TSLA", "INTC", "CMCSA", "CSCO", "NVDA", "PYPL",
                    "ADBE", "AMGN", "TXN", "AVGO", "GILD", "CHTR", "BIDU", "MU"]

    path = os.path.join(os.pardir, "data/")
    downlaod_stock_histories(path, tickers_list, period='1y')

    model = load_model("models/trained_model_acc_0.809393048286438.h5")
    for tick in tickers_list:

        data = get_one_stocks_data(tick)
        test_record = np.array([data[-1]])
        features = test_record[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]]
        # print(features[:, [0, 1]])
        labels = test_record[:, [test_record.shape[1] - 1]]
        features = features.astype(np.float32)
        pred = model.predict(features, verbose=0)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        # print("prediction: {} , truth: {}".format(pred[0], labels[-1]))
        print("{} Open:{} Close: {} prediction: {}".format(tick, features[:, [0]], features[:, [1]], pred[0]))

    # labels[labels == 'down'] = 0
    # labels[labels == 'up'] = 1

    features = features.astype(np.float32)
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


test_3()
