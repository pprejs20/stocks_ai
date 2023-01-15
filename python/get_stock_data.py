import os

import yfinance as yf
import pandas as pd
import matplotlib as plt
import numpy as np
from typing import *
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf

counter = 0

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
    global counter
    for tick in ticker_list:
        s_time = time.time()
        data = yf.Ticker(tick).history(period='max')
        data = data[data['Open'] != 0]
        a_path = path + "/individual_stock_data/" + tick + ".csv"
        print(a_path)
        counter += 1
        print(counter)
        try:

            # data.to_csv(path + "/individual_stock_data/" + tick + ".csv")
            data.to_csv(a_path)
            print("Time taken: {}s".format(time.time() - s_time))
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
        return data, files

    except FileNotFoundError:
        print("[Error] Path '{}' doesn't exist!")

def get_one_record(opening_prices):
    pass

print("Starting...")
folder = "data"
filename = "sp500_tickers.csv"
path = os.path.join(os.pardir, folder + "/" + filename)

# update_sp500_tickers(path, folder)
sp500_tickers = get_sp500_tickers(path)
data = pd.read_csv(os.pardir + "/data/individual_stock_data/XOM.csv")
data['gold_std'] = data['Open'].shift(-1)
true_data = data[['Open', 'gold_std']]
true_data = true_data[:-1]

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
print(true_data.columns.values)
true_data = true_data[20:]

# true_data = true_data[-2000:]

# print(true_data[:-1])
features = np.array(true_data[['Open', 'ma10', 'ma20', 'std20', 'upper_band', 'lower_band']])
labels = np.array(true_data['gold_std'])

# print(prices.reshape(-1, 1))
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
print(x_train.shape)
print(y_train.shape)

# model = LinearRegression()
# model.fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))
# predictions = model.predict(x_test.reshape(-1, 1))
# print(predictions.shape)
# results = pd.DataFrame({'preds': predictions.flatten(), 'truths': y_test})
# print(results)

print("Creating sequential model...")
# Create a sequential model
model = tf.keras.Sequential()

# Add a dense layer with input shape 1 and output shape 64
model.add(tf.keras.layers.Dense(64, input_shape=(6,), activation='relu'))

# Add a dense layer with output shape 64
model.add(tf.keras.layers.Dense(64, activation='relu'))

# Add an output layer with output shape 1
model.add(tf.keras.layers.Dense(1))
print("Compiling..")
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
print("Training...")
# Fit the model with the training data
model.fit(x_train, y_train, epochs=50, batch_size=32)
print("Done training ...")
# Make predictions on the test data
predictions = model.predict(x_test)
print("Loading up predictions!")

results = pd.DataFrame({'preds': predictions.flatten(), 'truths': y_test})
print(results)



mse = mean_squared_error(y_test.reshape(-1, 1), predictions)
r2 = r2_score(y_test.reshape(-1, 1), predictions)

print("Mean Squared Error: " + str(mse))
print("R2: " + str(r2))

# print("Predict: 108.470001220703")
# my_pred = model.predict([[y_test[0]]])
# print("prev prediction {}   ,   second pred: {} , original num: {}".format(predictions.flatten()[0], my_pred[0], y_test[0]))
# my_pred = model.predict([[35.188521]])
# print(my_pred)



# downlaod_stock_histories(os.path.join(os.pardir, "data"), sp500_tickers)
# histories, files = get_all_stocks_histories(os.pardir + "/data/")
# for i in range(0, len(histories)):
#     print(histories[i])
#     print(files[i])
# print(histories)


