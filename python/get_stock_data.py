import os

import yfinance as yf
import pandas as pd
import matplotlib as plt
import numpy as np
from typing import *
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def update_sp500_tickers(path: str, folder: str):
    sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    sp500_list = np.array(sp500[0]['Symbol'])
    try:
        with open(path, 'w') as file:
            for tick in sp500_list:
                file.write(tick + ",")

    except FileNotFoundError:
        os.mkdir(os.path.join(os.pardir, folder))
        update_sp500_tickers(folder, filename)


def get_sp500_tickers(path: str) -> List[str]:
    try:
        with open(path, 'r') as file:
            contents = file.readline()
            sp500_tickers = contents.split(',')
            return sp500_tickers
    except FileNotFoundError:
        print("[Error] List of S&P 500 Tickers has not been downloaded")


def downlaod_stock_histories(path: str, ticker_list: List[str]):
    for tick in ticker_list:
        data = yf.Ticker(tick).history(period='max')
        data = data[data['Open'] != 0]
        try:
            data.to_csv(path + "/individual_stock_data/" + tick + ".csv")
        except FileNotFoundError:
            os.mkdir(path + "/individual_stock_data")
            downlaod_stock_histories(path, ticker_list)


def get_all_stocks_histories(path: str):
    local_path = path + "/individual_stock_data/"
    data = []
    try:

        files = os.listdir(path + "/individual_stock_data/")
        for file in files:
            data.append(pd.read_csv(local_path + file))
        return data

    except FileNotFoundError:
        print("[Error] Path '{}' doesn't exist!")

def find_best_estimators(x_train, y_train, x_test, y_test):
    estimators = [5, 10, 25, 30, 50, 60, 75, 100]
    accuracies = []
    for est in estimators:
        clf1 = RandomForestClassifier(n_estimators=est, random_state=42)  # random_state=42

        # Fit the classifier to the training data
        clf1.fit(x_train, y_train)

        # Make predictions on the test data
        predictions = clf1.predict(x_test)
        accuracies.append(accuracy_score(y_test, predictions))

    best = max(accuracies)
    _index = accuracies.index(best)
    return accuracies[_index], estimators[_index]


print("Starting...")
folder = "data"
filename = "sp500_tickers.csv"
path = os.path.join(os.pardir, folder + "/" + filename)

# Un-comment this code to download latest training data
# update_sp500_tickers(path, folder)


sp500_tickers = get_sp500_tickers(path)
data = pd.read_csv(os.pardir + "/data/individual_stock_data/XOM.csv")
# data['gold_std'] = data['Open'].shift(-1)
# true_data = data[['Open', 'gold_std']]
true_data = data[['Open']]


# Assume that your DataFrame is called 'df'

# Calculate moving averages
true_data['ma10'] = true_data['Open'].rolling(window=10).mean()
true_data['ma20'] = true_data['Open'].rolling(window=20).mean()

# Calculate Bollinger Bands
true_data['std20'] = true_data['Open'].rolling(window=20).std()
true_data['upper_band'] = true_data['ma20'] + 2 * true_data['std20']
true_data['lower_band'] = true_data['ma20'] - 2 * true_data['std20']

# Calculate Relative Strength Index (RSI)
delta = true_data['Open'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
true_data['rsi'] = 100 - (100 / (1 + rs))
true_data = true_data[20:]
true_data['label'] = str('NaN')
np_array_data = np.array(true_data)
for i in range(len(np_array_data) - 1):
    if np_array_data[i][0] <= np_array_data[i+1][0]:
        np_array_data[i][-1] = "up"
    else:
        np_array_data[i][-1] = "down"

np_array_data = np_array_data[:-1]
labels = np_array_data[:, [7]]
features = np_array_data[:, range(0, 7)]
print(features)
print(labels)

# print(np_array_data[:-1, [0, 7]])
# df = pd.DataFrame({"Open": np_array_data[:, [0]],
#                    "ma10": np_array_data[:, [1]],
#                    "ma20": np_array_data[:, [2]],
#                    "std20": np_array_data[:, [3]],
#                    "upper_band": np_array_data[:, [4]],
#                    "lower_band": np_array_data[:, [5]],
#                    "rsi": np_array_data[:, [6]],
#                    "label": np_array_data[:, [7]]
#                    })
#

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

#Create an instance of the classifier
# clf1 = RandomForestClassifier(n_estimators=100) #random_state=42
#
# #Fit the classifier to the training data
# clf1.fit(x_train, y_train)
#
# #Make predictions on the test data
# predictions = clf1.predict(x_test)
#
# #Evaluate the performance of the classifier
# print("Random Forest Accuracy:", accuracy_score(y_test, predictions))
# print("Random State: ", clf1.random_state)

accuracy, estimators = find_best_estimators(x_train, y_train, x_test, y_test)
print("Best accuracy: ", accuracy)
print("Best estimators: ", estimators)

