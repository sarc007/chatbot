# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Use a breakpoint in the code line below to debug your script.
    # print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

    print(tf.__version__)
    physical_devices = tf.config.list_physical_devices("GPU")
    print(physical_devices)
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape, y_train.shape)
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # for i in range(6):
    #     plt.subplot(2, 3, i+1)
    #     plt.imshow(x_train[i], cmap='gray')
    # plt.show()
    # model
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10),
    ])
    print(model.summary())
    #     loss and optimizer
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optim = keras.optimizers.Adam(learning_rate=0.001)
    metrics = ["accuracy"]
    model.compile(loss=loss, optimizer=optim, metrics=metrics)

    # training
    batch_size = 64
    epochs = 5

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)
    # evaluate
    model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)

    # predictions
    probability_model = keras.models.Sequential([
        model,
        keras.layers.Softmax(),
    ])
    predictions = probability_model(x_test)
    pred0 = predictions[0]
    print(pred0)
    label0 = np.argmax(pred0)
    print(label0)
# model + softmax
    predictions1 = model(x_test)
    predictions1 = tf.nn.softmax(predictions1)
    pred1 = predictions[0]
    print(pred0)
    label1 = np.argmax(pred1)
    print(label1)
#  another way to predict
    predictions2 = model.predict(x_test, batch_size=batch_size)
    predictions2 = tf.nn.softmax(predictions2)
    pred2 = predictions2[0]
    print(pred2)
    label2 = np.argmax(pred2)
    print(label2)
    pred5 = predictions[0:5]
    print(pred5.shape)
    print(pred5)
    label5 = np.argmax(pred5, axis=1)
    print(label5)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
