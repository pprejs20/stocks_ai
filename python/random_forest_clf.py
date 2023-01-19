import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from python.get_stock_data import get_sp500_tickers, generate_and_save_all_training_data

print("Starting...")
folder = "data"
filename = "sp500_tickers.csv"
path = os.path.join(os.pardir, folder + "/" + filename)
sp500_tickers = get_sp500_tickers(path)

# Un-comment this code to download latest training data
# update_sp500_tickers(path, folder)

# Un-comment this code to regenerate the training data, if any data has been updated
# generate_and_save_all_training_data(sp500_tickers, "full_training_data.csv")

data = pd.read_csv("full_training_data.csv")
data = data.dropna()
data = np.array(data)
data = data[:, [1, 2, 3, 4, 5, 6, 7, 8]]

print(data)
print(data.shape)

labels = data[:, [7]]
features = data[:, range(0, 7)]
# print(features)
# print(labels)

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
clf1 = RandomForestClassifier(n_estimators=100) #random_state=42

#Fit the classifier to the training data
clf1.fit(x_train, y_train)

#Make predictions on the test data
predictions = clf1.predict(x_test)

#Evaluate the performance of the classifier
print("Random Forest Accuracy:", accuracy_score(y_test, predictions))
# print("Random State: ", clf1.random_state)

# accuracy, estimators = find_best_estimators(x_train, y_train, x_test, y_test)
# print("Best accuracy: ", accuracy)
# print("Best estimators: ", estimators)
