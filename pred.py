import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import load_model

model = load_model("data/myModel_stock_LSTM.h5")

# 假設你已經有了訓練資料，並且放在名為"data.csv"的CSV檔案中
data = pd.read_csv("data/stock_google_plus.csv")

# 假設你的資料包括一列為"Close"的收盤價
close_prices = data["Close"].values.reshape(-1, 1)

# 將收盤價歸一化到0到1的範圍內
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close_prices = scaler.fit_transform(close_prices)

# 假設你的模型是根據過去N天的收盤價預測未來一天的收盤價
# 這裡需要根據你的模型設計對數據進行適當的轉換
# 假設N=60，即使用過去60天的收盤價作為特徵進行預測
N = 60
X_test = []
for i in range(N, len(scaled_close_prices)):
    X_test.append(scaled_close_prices[i-N:i, 0])
X_test = np.array(X_test)

# 使X_test具有LSTM模型所期望的形狀 (samples, time steps, features)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 使用模型進行預測
predicted_prices = model.predict(X_test)

# 將預測結果反歸一化
predicted_prices = scaler.inverse_transform(predicted_prices)

# 假設你想獲取第一個預測的價格
first_predicted_price = predicted_prices[0][0]
print("First predicted price:", first_predicted_price)
