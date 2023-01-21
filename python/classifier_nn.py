from keras import activations

from get_stock_data import *
from keras.layers import Dense, BatchNormalization
from keras.models import Sequential

sp500_tickers = get_sp500_tickers()
#
# path = os.path.join(os.pardir, "data/")
# downlaod_stock_histories(path, sp500_tickers)

# generate_and_save_all_training_data(sp500_tickers, filename="full_training_data3.csv")

print("Starting...")

data = pd.read_csv("full_training_data3.csv")
data = data.dropna()
data = np.array(data)
data = data[:, range(1, data.shape[1])]

print(data)
print(data.shape)


labels = data[:, [data.shape[1] - 1]]
features = data[:, range(0, data.shape[1] - 1)]
counter = 0
print(labels)
# for label in labels:
#     if label[0] == 'down':
#         counter += 1
# print(counter)
# exit()

# Convert lables to integers
labels[labels == 'down'] = 0
labels[labels == 'up'] = 1
labels = labels.astype(int)
print(labels)
features = features.astype(np.float32)
print(features)
print(np.sum(labels == 1))

x_train, x_test, y_train, y_test = train_test_split(features, labels.ravel(), test_size=0.2, random_state=42)
joined_array = np.column_stack((x_test, y_test))
np.savetxt("testing_data.csv", joined_array, delimiter=',')


# Original hidden layer:
# model.add(Dense(16, activation=activations.relu))

# model = Sequential()
# model.add(Dense(16, input_dim=features.shape[1], activation='relu'))
# model.add(Dense(8, activation=activations.relu))
# # model.add(Dense(16, activation=activations.leaky_relu))
# model.add(Dense(1, activation='sigmoid'))
#
# # Compile the model
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model = Sequential()
model.add(Dense(16, input_dim=features.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dense(8, activation=activations.relu))
model.add(BatchNormalization())
# model.add(Dense(16, activation=activations.leaky_relu))
model.add(Dense(1, activation='sigmoid'))
model.add(BatchNormalization())

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Original line of code
# model.fit(x_train, y_train, epochs=10, batch_size=32)

model.fit(x_train, y_train, epochs=50, batch_size=16)
score = model.evaluate(x_test, y_test, verbose=1)
model.save("models/trained_model_acc_{}.h5".format(score[1]))
print("Loss: {} , Accuracy: {}".format(score[0], score[1]))
