import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers
from keras.layers import preprocessing, Normalization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    print(tf.__version__)
    physical_devices = tf.config.list_physical_devices("GPU")
    # print(physical_devices)
    dataset = pd.read_csv('.\\data\\deg-fah\\degfah.csv')
    # print(dataset)
    train_ds = dataset.sample(frac=1, random_state=0)
    test_ds = dataset.drop(train_ds.index)
    # print("--------------------Train DS ------------------------------")
    # print(train_ds)
    # print("--------------------Test DS ------------------------------")
    # print(test_ds)
    # print(train_ds.describe().transpose()[['mean', 'std']])
    (x_train, y_train) = train_ds["Fahrenheit"], train_ds["Degree Celsius"]
    (x_test, y_test) = test_ds["Fahrenheit"], test_ds["Degree Celsius"]
    # x_train, x_test = x_train / 255.0, x_test / 255.0

    print(x_train)
    print(y_train)
    lr = 0.1
    epoch = 500
    model = keras.models.Sequential([
        keras.layers.Dense(units=1, input_shape=[1]),
        # keras.layers.Dense(units=4, activation='tanh'),
        # keras.layers.Dense(units=2, activation='relu'),
        # keras.layers.Dense(units=1, activation='relu'),
    ])
    print(model.summary())
    model.compile(loss='mean_squared_error', optimizer=tf.optimizers.Adam(lr))
    trained_model = model.fit(train_ds["Fahrenheit"], train_ds["Degree Celsius"], epochs=epoch, verbose=False)
    loss = trained_model.history['loss']
    epoch_label = [x for x in range(epoch)]
    loss_df = pd.DataFrame(list(zip(loss, epoch_label)), columns=['Loss', 'Epoch'])
    sns.set(style='darkgrid')
    sns.lineplot(x='Epoch', y='Loss', data=loss_df)
    plt.show()
    degree_cel = -73.333
    print(f' {degree_cel} Degree Celcius to Fahrenheit  {model.predict([degree_cel])}')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
