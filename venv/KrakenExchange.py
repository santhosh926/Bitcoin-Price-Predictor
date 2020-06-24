import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

# deep learning library to build ml models
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Setting up the data frame
df = pd.read_csv('Kraken_BTCUSD_d.csv', header=0)
data = df.iloc[::-1]
df.head()

# Line Graph for the real data set
plt.plot(data['Date'], data['Close'])
plt.xticks(data['Date'][::365])
plt.xticks(rotation=45)
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.title('Kraken BTC-USD Exchange Prices Over Time')
plt.show()

price_data = data.iloc[:, 5:6]  # getting only the close price data
scaler = MinMaxScaler()  #
# scales the data points to much smaller values so they can easily be used in a Keras model
price_data = scaler.fit_transform(price_data)

# splitting the data (80% for training and 20% for testing)
split = int(len(price_data) * 0.8)
train_set = price_data[:split]
test_set = price_data[split:]

x_train = train_set[0:len(train_set)-1]  # input data
y_train = train_set[1:len(train_set)]    # target data

x_test = test_set[0:len(test_set)-1]
y_test = test_set[1:len(test_set)]

# reshaping input data into a numpy array that can be used in our model
x_train = np.reshape(x_train, (len(x_train), 1, x_train.shape[1]))
x_test = np.reshape(x_test, (len(x_test), 1, x_test.shape[1]))

model = Sequential()  # initialize a sequential keras model
# add an LSTM Layer to the model (input)
model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(LSTM(128))  # add a second LSTM layer from the hidden sequences of the first layer (for the output)
model.add(Dense(1))   # followed by a dense layer to make sure there is a 1-dimensional output

# compile model using an adam optimizer
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, shuffle=False)  # fits the model to the data, with a set number of iterations

predicted_price = model.predict(x_test)  # Generates output predictions for the input samples in a numpy array.
predicted_price = scaler.inverse_transform(predicted_price)  # scales the predicted data back up
real_price = scaler.inverse_transform(y_test)   # scales the real data back up

# Line Graph to show the predicted price vs the real price
plt.plot(predicted_price, color='blue', label='Predicted Price of Bitcoin on Kraken Exchange')
plt.plot(real_price, color='red', label='Real Price of Bitcoin on Kraken Exchange')
plt.title('Real vs. Predicted Kraken BTC-USD Exchange Prices Over Time')
plt.xlabel('Time (Days)')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
