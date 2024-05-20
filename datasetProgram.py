import pandas as pd
import numpy as np


def calculate_sma(data, window):
    return data.rolling(window).mean()


def calculate_ema(data, window):
    return data.ewm(span=window, adjust=False).mean()


def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(data, fast_window=12, slow_window=26, signal_window=9):
    ema_fast = calculate_ema(data, fast_window)
    ema_slow = calculate_ema(data, slow_window)
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, macd_signal


def calculate_technical_indicator(df):
    df['SMA_5'] = calculate_sma(df['Close'], 5)
    df['SMA_20'] = calculate_sma(df['Close'], 20)
    df['EMA_5'] = calculate_ema(df['Close'], 5)
    df['EMA_20'] = calculate_ema(df['Close'], 20)
    df['MACD'], df['MACD_Signal'] = calculate_macd(df['Close'])
    return df


if __name__ == '__main__':
    df = pd.read_csv('data/stock_google.csv')
    df = calculate_technical_indicator(df)
    df.to_csv('data/stock_google_new.csv', index=False)
    print('Finished calculating technical indicator and output!')
