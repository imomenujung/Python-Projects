import streamlit as st
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import SimpleRNN, GRU, LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error

# Fungsi untuk mengambil data
def get_train_test(url, split_percent=0.8):
    data = pd.read_csv(url)
    data = data.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    split = int(len(data_scaled) * split_percent)
    train_data, test_data = data_scaled[:split], data_scaled[split:]
    
    return train_data, test_data, data, scaler

# Fungsi untuk menyiapkan dataset

def get_XY(data, time_steps):
    X, Y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        Y.append(data[i + time_steps, 0])
    return np.array(X).reshape(-1, time_steps, 1), np.array(Y)

# Fungsi untuk membuat model
def create_model(model_type, hidden_units, dense_units, input_shape, activation):
    model = Sequential()
    if model_type == "SimpleRNN":
        model.add(SimpleRNN(hidden_units, activation=activation, input_shape=input_shape))
    elif model_type == "LSTM":
        model.add(LSTM(hidden_units, activation=activation, input_shape=input_shape))
    elif model_type == "GRU":
        model.add(GRU(hidden_units, activation=activation, input_shape=input_shape))
    model.add(Dense(dense_units))
    model.compile(optimizer='adam', loss='mse')
    return model

# Fungsi untuk plotting hasil
import matplotlib.pyplot as plt

def plot_result(true_train, true_test, pred_train, pred_test):
    plt.figure(figsize=(12,6))
    plt.plot(np.arange(len(true_train)), true_train, label='Actual Train')
    plt.plot(np.arange(len(true_train), len(true_train) + len(true_test)), true_test, label='Actual Test')
    plt.plot(np.arange(len(pred_train)), pred_train, '--', label='Predicted Train')
    plt.plot(np.arange(len(pred_train), len(pred_train) + len(pred_test)), pred_test, '--', label='Predicted Test')
    plt.legend()
    st.pyplot(plt)

# Streamlit UI
st.title("Hyperparameter Tuning - RNN, LSTM, GRU")

# Parameter tuning
model_type = st.selectbox("Model Type", ["SimpleRNN", "LSTM", "GRU"])
activation = st.selectbox("Activation Function", ['tanh', 'sigmoid', 'relu'])
hidden_units = st.slider("Hidden Units", 10, 100, 50)
epochs = st.slider("Epochs", 10, 200, 75)
batch_size = st.slider("Batch Size", 16, 128, 32)
verbose = st.selectbox("Verbose", [0, 1, 2])

# Load data
time_steps = 5
train_data, test_data, data, scaler = get_train_test("https://raw.githubusercontent.com/imomenujung/MPDW/main/Tugas%20Akhir/data176.csv")
trainX, trainY = get_XY(train_data, time_steps)
testX, testY = get_XY(test_data, time_steps)

# Train model
if st.button("Train Model"):
    model = create_model(model_type, hidden_units, 1, (time_steps, 1), activation)
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=verbose)
    train_predict = model.predict(trainX)
    test_predict = model.predict(testX)
    
    MAPE_train = mean_absolute_percentage_error(scaler.inverse_transform(trainY.reshape(-1,1)), scaler.inverse_transform(train_predict))
    MAPE_test = mean_absolute_percentage_error(scaler.inverse_transform(testY.reshape(-1,1)), scaler.inverse_transform(test_predict.reshape(-1,1)))
    
    st.write(f"MAPE Train: {round(MAPE_train * 100,2)}%")
    st.write(f"MAPE Test: {round(MAPE_test * 100,2)}%")
    
    plot_result(scaler.inverse_transform(trainY.reshape(-1,1)), scaler.inverse_transform(testY.reshape(-1,1)), scaler.inverse_transform(train_predict.reshape(-1,1)), scaler.inverse_transform(test_predict.reshape(-1,1)))

# AutoTune
def auto_tune():
    best_mape = float("inf")
    best_params = {}
    
    for model_t in ["SimpleRNN", "LSTM", "GRU"]:
        for units in [10, 50, 100]:
            for act in ['tanh', 'sigmoid', 'relu']:
                model = create_model(model_t, units, 1, (time_steps, 1), act)
                model.fit(trainX, trainY, epochs=50, batch_size=32, verbose=0)
                test_predict = model.predict(testX)
                mape_test = mean_absolute_percentage_error(scaler.inverse_transform(testY.reshape(-1,1)), scaler.inverse_transform(test_predict.reshape(-1,1)))
                
                if mape_test < best_mape:
                    best_mape = mape_test
                    best_params = {"model_type": model_t, "hidden_units": units, "activation": act, "MAPE": best_mape}
    
    return best_params

if st.button("Auto Tune"):
    best_params = auto_tune()
    st.write(f"Best Parameters: {best_params}")
