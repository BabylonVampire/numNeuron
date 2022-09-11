import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt
import os
import cv2

def convert_pic(adress):
    img = cv2.imread(adress)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.bitwise_not(img)

#ОБНАРУЖЕНИЕ КРАЕВ ИЗОБРАЖЕНИЯ

    h1 = h2 = w1 = w2 = 0

    for i in range(np.shape(img)[0]):
        if img[i].any():
            h1 = i
            break

    for i in reversed(range(np.shape(img)[0])):
        if img[i].any():
            h2 = i
            break

    for i in range(np.shape(img)[0]):
        if img[:, i].any():
            w1 = i
            break

    for i in reversed(range(np.shape(img)[0])):
        if img[:, i].any():
            w2 = i
            break

    c = min(h1, w1)

    # ОБРЕЗАНИЕ ИЗОБРАЖЕНИЯ ПО КРАЯМ
    img = img[(h1 - c):(h2 + c), (w1 - c):(w2 + c)]

    plt.imshow(img)
    plt.show()

    img = cv2.resize(img, (28, 28))

    img = (np.ceil(img / 255.0) * 255.0).astype(np.uint8)
    img = img / 255.0
    #Перевод в поддерживаемый тип
    img = img.astype(np.float32)
    return img

def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train / 255
    x_test = x_test / 255

    model = keras.Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(100, activation='relu'),
        Dense(10, activation='softmax')])

    y_train_cat = keras.utils.to_categorical(y_train, 10)
    y_test_cat = keras.utils.to_categorical(y_test, 10)

    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train_cat, batch_size=32, epochs=9, validation_split=0.2)

    for file in os.scandir('tests'):
        if not file.is_file():
            continue

        img = convert_pic(file.path)
        argument = np.expand_dims(img, axis=0)
        result = model.predict(argument)

        print(np.argmax(result))

        plt.imshow(img)
        plt.show()

if __name__ == '__main__':
    main()