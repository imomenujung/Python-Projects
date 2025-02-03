from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, LSTM, GRU
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_percentage_error
from datetime import datetime, date
import seaborn as sns

# Membaca Data
df = pd.read_csv("https://raw.githubusercontent.com/imomenujung/MPDW/main/Tugas%20Akhir/data176.csv", sep=",")
df.head()

def get_train_test(url, split_percent=0.8):
    global scaler
    df = pd.read_csv(url)[["Average_Kurs"]]
    data = np.array(df.values.astype('float32'))
    n = len(data)
    # Point for splitting data into train and test
    scaler = MinMaxScaler(feature_range=(0, 1))
    split = int(n*split_percent)
    train_data = data[range(split)]
    test_data = data[split:]
    train_data = scaler.fit_transform(train_data).flatten()
    test_data = scaler.transform(test_data).flatten()

    return train_data, test_data, data

def create_RNN(hidden_units, dense_units, input_shape, activation):
    model = Sequential()
    model.add(SimpleRNN(hidden_units, input_shape=input_shape,
                        activation=activation[0]))
    model.add(Dense(units=dense_units, activation=activation[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def get_XY(dat, time_steps):
    X, Y = [], []
    for i in range(len(dat) - time_steps):
        X.append(dat[i:i+time_steps])  # Mengambil window dari data
        Y.append(dat[i+time_steps])    # Target adalah nilai setelah window
    X, Y = np.array(X), np.array(Y)
    X = np.reshape(X, (X.shape[0], time_steps, 1))  # Reshape ke format RNN
    return X, Y

def plot_result(trainY, testY, train_predict, test_predict):
    actual = np.append(trainY, testY)
    predictions = np.append(train_predict, test_predict)
    rows = len(actual)
    plt.figure(figsize=(15, 6), dpi=80)
    plt.plot(range(rows), actual)
    plt.plot(range(rows), predictions)
    plt.axvline(x=len(trainY), color='r')
    plt.legend(['Data Train', 'Data Test'])
    plt.xlabel('Period (Week)')
    plt.ylabel('Value of "Rupiah Exchange Rate Against USD"')

#EPOCH 100
# Metode = Simple RNN
# Activation = "tanh", pilihannnya : 'tanh','sigmoid','relu'
# Nilai yang dapat diubah : hidden_units=50, epochs=100, batch_size=32,verbose=2
train_data, test_data, data = get_train_test("https://raw.githubusercontent.com/imomenujung/MPDW/main/Tugas%20Akhir/data176.csv",split_percent=0.8)
trainX, trainY = get_XY(train_data, time_steps)
testX, testY = get_XY(test_data, time_steps)

# Pemodelan
model = create_RNN(hidden_units=50, dense_units=1, input_shape=(time_steps,1),
                   activation=['tanh', 'tanh'])
model.fit(trainX, trainY, epochs=100, batch_size=32, verbose=2)

# Prediksi Data
train_predict = model.predict(trainX)
test_predict = model.predict(testX)

# Evaluasi Model
MAPE_train_RNN = mean_absolute_percentage_error(scaler.inverse_transform(trainY.reshape(-1,1)), scaler.inverse_transform(train_predict))
MAPE_test_RNN = mean_absolute_percentage_error(scaler.inverse_transform(testY.reshape(-1,1)), scaler.inverse_transform(test_predict.reshape(-1,1)))

print(f"MAPE_train_RNN = {round(MAPE_train_RNN * 100,2)} %")
print(f"MAPE_test_RNN = {round(MAPE_test_RNN * 100,2)} %")

plot_result(scaler.inverse_transform(trainY.reshape(-1,1)), scaler.inverse_transform(testY.reshape(-1,1)), scaler.inverse_transform(train_predict.reshape(-1,1)), scaler.inverse_transform(test_predict.reshape(-1,1)))
