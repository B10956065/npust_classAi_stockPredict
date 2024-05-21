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
    讀取資料集並進行一些必要的處理。他喵的給我加註解啊啊啊啊啊!!!!!!!!!!!!!
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
    """
    把資料轉成LSTM合適的格式。chatgpt說的
    :param data: 資料
    :param time_step: 我不知道
    :return: 我不知道
    """
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)


def train_init_model(X_train, y_train, time_step=60, epochs=1, batch_size=1):
    """
    訓練初始模型。
    :param X_train: 訓練資料集
    :param y_train: 為什麼要把資料集切二份，請去問簽名的那個人
    :param time_step: 同上
    :param epochs: 同上
    :param batch_size: 同上
    :return: 訓練好的AI模型
    """
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
    :param model: old model
    :param new_data: new data
    :param time_step: ???
    :param epochs: ???
    :param batch_size: ???
    :return:新模型
    """
    X, y = create_dataset(new_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    return model


# main predict
def predict_next_day(model, data, scaler, time_step=60):
    """
    Predict next day's stock's price.
    :param model: model
    :param data: full-data
    :param scaler: model-2
    :param time_step: how long should short time memory should look back
    :return: next day's price. It should be a float.
    """
    last_60_days = data[-time_step:]
    last_60_days_scaled = last_60_days.reshape(-1, 1)

    # Reshape input to be [samples, time steps, features]
    X_test = last_60_days_scaled.reshape(1, time_step, 1)

    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)

    return pred_price[0, 0]


# 畫圖而已，不重要
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
