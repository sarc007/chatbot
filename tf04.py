import os

# https://www.youtube.com/watch?v=g2t7M2HTZ9U&list=PLqnslRFeH2Uqfv1Vz3DqeQfy0w20ldbaV&index=4

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.layers import preprocessing, Normalization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)


def main():
    print(tf.__version__)
    physical_devices = tf.config.list_physical_devices("GPU")
    print(physical_devices)

    url = 'https://archive.ics.uci.edu/ml/datasets/auto+mpg'
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower',
                    'Weight', 'Acceleration', 'Model Year', 'Origin']

    dataset = pd.read_csv('.\\data\\auto-mpg\\auto-mpg.data-original.csv', names=column_names,
                          na_values='?', comment='\t', sep=' ', skipinitialspace=True)
    dataset = dataset.dropna()
    origin = dataset.pop('Origin')
    dataset['USA'] = (origin == 1) * 1
    dataset['Europe'] = (origin == 2) * 1
    dataset['Japan'] = (origin == 3) * 1

    print(dataset.tail())

    train_ds = dataset.sample(frac=0.8, random_state=0)
    test_ds = dataset.drop(train_ds.index)
    print(dataset.shape, train_ds.shape, test_ds.shape)
    print(train_ds.describe().transpose())

    train_features = train_ds.copy()
    test_features = test_ds.copy()

    train_labels = train_features.pop('MPG')
    test_labels = test_features.pop('MPG')
    plot('Horsepower', train_features=train_features, train_labels=train_labels)
    plot('Weight', train_features=train_features, train_labels=train_labels)
    print(train_ds.describe().transpose()[['mean', 'std']])
    normalizer = Normalization()
    normalizer.adapt(np.array(train_features))
    print(normalizer.mean.numpy())
    first = np.array(train_features[:1])
    print('first example : ', first)
    print('Normalized example : ', normalizer(first).numpy())

def plot(feature, x=None, y=None, train_features=None, train_labels=None):
    plt.figure(figsize=(10, 8))
    plt.scatter(train_features[feature], train_labels, label='Data')
    if x is not None and y is not None:
        plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel(feature)
    plt.ylabel('MPG')
    plt.legend()
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
