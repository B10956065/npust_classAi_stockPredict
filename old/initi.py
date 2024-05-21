import time
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.models import load_model


def read_and_preprocess(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path)

    # Ensure the data is sorted by date
    data = data.sort_values('Date')

    # Use only the 'Close' prices for simplicity
    prices = data['Close'].values

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices.reshape(-1, 1))

    return scaled_prices, scaler


def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)


def train_init_model(data, time_step=60, epochs=1, batch_size=1):
    X, y = create_dataset(data, time_step)

    # Reshape input to be [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Create the LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)

    return model


def incremental_learning(model, new_data, time_step=60, epochs=1, batch_size=1):
    X, y = create_dataset(new_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
    return model


def predict_next_day(model, data, time_step=60):
    last_60_days = data[-time_step:]
    last_60_days_scaled = last_60_days.reshape(-1, 1)

    # Reshape input to be [samples, time steps, features]
    X_test = last_60_days_scaled.reshape(1, time_step, 1)

    pred_price = model.predict(X_test)
    pred_price = scaler.inverse_transform(pred_price)
    return pred_price[0, 0]


if __name__ == '__main__':
    scaled_prices, scaler = read_and_preprocess('../data/stock_google_new.csv')
    time_step = 60
    initial_model = train_init_model(scaled_prices, time_step)
    predicted_price = predict_next_day(initial_model, scaled_prices)
    print(f"Predicted next day's price: {predicted_price}")
