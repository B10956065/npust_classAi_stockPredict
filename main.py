# -*- coding: utf-8 -*-
import math
import time
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Input
from keras.callbacks import EarlyStopping


def read_and_preprocess(file_path):
    """

    :param file_path:The dataset.csv file path
    :return: data:The original dataset.
    scaled_prices:The scaled prices.
    scaler:The scaler
    """
    # Read the CSV file
    data = pd.read_csv(file_path)

    # Ensure the data is sorted by date
    data = data.sort_values('Date')
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Use only the 'Close' prices for simplicity
    prices = data['Close'].values

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices.reshape(-1, 1))

    return data, scaled_prices, scaler


def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)


def train_init_model(X_train, y_train, time_step=60, epochs=1, batch_size=1):
    # Reshape input to be [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # Create the LSTM model
    model = Sequential()
    model.add(Input(shape=(time_step, 1)))
    model.add(LSTM(units=100, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))

    # Show the summary of model
    model.summary()

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # EarlyStopping
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, callbacks=[es])

    return model


# incremental learning
def incremental_learning(model, new_data, time_step=60, epochs=1, batch_size=1):
    """
    input old model and new data, return new model
    :param model:
    :param new_data:
    :param time_step:
    :param epochs:
    :param batch_size:
    :return:
    """
    X, y = create_dataset(new_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    return model


# main predict
def predict_next_day(model, data, scaler, time_step=60):
    last_60_days = data[-time_step:]
    last_60_days_scaled = last_60_days.reshape(-1, 1)

    # Reshape input to be [samples, time steps, features]
    X_test = last_60_days_scaled.reshape(1, time_step, 1)

    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)

    return pred_price[0, 0]


# 畫圖而已
def plot_predictions(train_data, test_data, predicted_prices, time_step=60):
    plt.figure(figsize=(12, 6))
    # Plot the actual prices
    # plt.plot(train_data.index, train_data['Close'], label='Train Price')
    plt.plot(test_data.index, test_data['Close'], label='Test Price')
    # Plot the predicted prices
    plt.plot(test_data.index, predicted_prices, label='Predicted Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    _, scaled_prices, scaler = read_and_preprocess('data/stock_google_new.csv')
    joblib.dump(scaler, 'scaler.pkl')
