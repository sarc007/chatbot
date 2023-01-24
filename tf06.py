import datetime
import math
import time

import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yahooFinance
from keras.callbacks import EarlyStopping
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import tensorflow as tf
import pandas as pd
import seaborn as sns

# from datetime import datetime

# startDate , as per our convenience we can modify
startDate = datetime.datetime(2018, 1, 1)
# endDate , as per our convenience we can modify
endDate = datetime.datetime(2022, 6, 16)
stock_name = 'GOOGL'


# pass the parameters as the taken dates for start and end
# google_data = GetGoogleInformation.history(start=startDate, end=endDate)
#
# google_data = google_data.drop(['Dividends', 'Stock Splits'], axis=1)
# print(google_data.head())
# print(GetGoogleInformation.history(start=startDate, end=endDate))
# aapl = data.DataReader("AAPL",
#                        start='2015-1-1',
#                        end='2015-12-31',
#                        data_source='yahoo')['Open']
# print(aapl.head())


def get_stock_data(stock_name_, normalized=0):
    # aapl = data.DataReader("AAPL",
    #                        start='2015-1-1',
    #                        end='2015-12-31',
    #                        data_source='yahoo')['Adj Close']
    # url = "http://www.google.com/finance/historical?q=" + \
    #       stock_name_ + "&startdate=Jul+12%2C+2013&enddate=Jul+11%2C+2017&num=30&ei=rCtlWZGSFN3KsQHwrqWQCw&output=csv"
    #     url="http://www.google.com/finance/historical?q=%s&ei=u-lHWfGPNNWIsgHHqIqICw&output=csv" % stock_name
    # col_names = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    # stocks = pd.read_csv(GetGoogleInformation.history(start=startDate, end=endDate), header=0, names=col_names)
    GetGoogleInformation = yahooFinance.Ticker(stock_name)
    df_ = GetGoogleInformation.history(start=startDate, end=endDate)
    today = datetime.date.today()
    file_name = '.\\data\\stocks\\' + stock_name + '_stock_%s.csv' % today
    print(df_.shape)
    df_.to_csv(file_name)
    df_.drop(['Dividends', 'Stock Splits', 'Volume', 'Low'], axis=1, inplace=True)
    return df_


df = get_stock_data(stock_name, 0)
print(df.tail())

normalizing_factor = (max(df['High'])) + 1
df['High'] = df['High'] / normalizing_factor
df['Open'] = df['Open'] / normalizing_factor
df['Close'] = df['Close'] / normalizing_factor
print(df.head(5))


def load_data(stock, seq_len, test_train_split):
    amount_of_features = len(stock.columns)
    data_ = stock.to_numpy()  # pd.DataFrame(stock)
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data_) - sequence_length):
        result.append(data_[index: index + sequence_length])

    result = np.array(result)
    row = round(test_train_split * result.shape[0])
    print(f'row={row}')
    train = result[:int(row), :]
    x_train = train[:, :-1]
    y_train = train[:, -1][:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))

    return [x_train, y_train, x_test, y_test]


def build_model(layers):
    model = Sequential()

    model.add(LSTM(
        input_dim=layers[0],
        output_dim=layers[1],
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        output_dim=layers[2]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop", metrics=['accuracy'])
    print("Compilation Time : ", time.time() - start)
    return model


def build_model2(layers):
    d = 0.2
    model = Sequential()
    model.add(LSTM(128, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
    model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))
    model.add(Dense(16, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='glorot_uniform', activation='relu'))
    model.compile(loss='mean_squared_error', optimizer=tf.optimizers.Adam(lr) , metrics=['accuracy'])
    return model


window = 25
X_train, y_train, X_test, y_test = load_data(df[::-1], window, 0.7)
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)
lr = 0.001
# model = build_model([3,lag,1])
model = build_model2([3, window, 1])

# model.fit(
#     X_train,
#     y_train,
#     batch_size=64,
#     epochs=500,
#     validation_split=0.1,
#     verbose=1)
#     # callbacks=[early_stopping])
epoch = 500
early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
trained_model = model.fit(
    X_train,
    y_train,
    batch_size=100,
    epochs=epoch,
    validation_split=0.1,
    verbose=1,
    callbacks=[early_stopping])
loss = trained_model.history['loss']
epoch_label = [x for x in range(epoch)]
loss_df = pd.DataFrame(list(zip(loss, epoch_label)), columns=['Loss', 'Epoch'])
sns.set(style='darkgrid')
sns.lineplot(x='Epoch', y='Loss', data=loss_df)
plt.show()
trainScore = model.evaluate(X_train, y_train, verbose=0)
print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

# print(X_test[-1])
diff = []
ratio = []
p = model.predict(X_test)
for u in range(len(y_test)):
    pr = p[u][0]
    ratio.append((y_test[u] / pr) - 1)
    diff.append(abs(y_test[u] - pr))
    # print(u, y_test[u], pr, (y_test[u]/pr)-1, abs(y_test[u]- pr))

plt2.plot(p, color='red', label='prediction')
plt2.plot(y_test, color='blue', label='y_test')
plt2.legend(loc='upper left')
plt2.show()

