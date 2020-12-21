from interface import Ui_MainWidget
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from keras.datasets import cifar10
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.models import Sequential
from keras import optimizers
from keras.metrics import categorical_crossentropy
from keras.utils import np_utils,plot_model
import keras.models as km


def change_tag_to_string(input):
    a = input[0]
    string = ''
    if a == 0:
        string = 'airplane'
    elif a == 1:
        string = 'automobile'
    elif a == 2:
        string = 'bird'
    elif a == 3:
        string = 'cat'
    elif a == 4:
        string = 'deer'
    elif a == 5:
        string = 'dog'
    elif a == 6:
        string = 'frog'
    elif a == 7:
        string = 'horse'
    elif a == 8:
        string = 'ship'
    elif a == 9:
        string = 'truck'
    else:
        string = 'error'
    return string


class MainUi(QtWidgets.QWidget, Ui_MainWidget):
    def __init__(self, parent=None):
        super(MainUi, self).__init__(parent)
        self.setupUi(self)
        self.btn1.clicked.connect(self.func1)
        self.btn2.clicked.connect(self.func2)
        self.btn3.clicked.connect(self.func3)
        self.btn4.clicked.connect(self.func4)
        self.btn5.clicked.connect(self.func5)

        self.bool_local_train = False
        # True -> train in local computer(very slow in my case)
        # False -> use saved model './Hw1_5_model.h5', ran on google colab
        self.batch_size = 50
        self.learning_rate = 0.01

        (trainX, self.trainY), (testX, testY) = cifar10.load_data()
        self.x_train = trainX.astype('float32')/255
        self.x_test = testX.astype('float32')/255
        self.y_train = np_utils.to_categorical(self.trainY)
        self.y_test = np_utils.to_categorical(testY)

        vgg16_layers = [
            # block1
            Conv2D(64, kernel_size=[3, 3], padding='same', activation='relu'),
            Conv2D(64, kernel_size=[3, 3], padding='same', activation='relu'),
            MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
            # block2
            Conv2D(128, kernel_size=[3, 3], padding='same', activation='relu'),
            Conv2D(128, kernel_size=[3, 3], padding='same', activation='relu'),
            MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
            # block3
            Conv2D(256, kernel_size=[3, 3], padding='same', activation='relu'),
            Conv2D(256, kernel_size=[3, 3], padding='same', activation='relu'),
            Conv2D(256, kernel_size=[3, 3], padding='same', activation='relu'),
            MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
            # block4
            Conv2D(512, kernel_size=[3, 3], padding='same', activation='relu'),
            Conv2D(512, kernel_size=[3, 3], padding='same', activation='relu'),
            Conv2D(512, kernel_size=[3, 3], padding='same', activation='relu'),
            MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
            # block5
            Conv2D(512, kernel_size=[3, 3], padding='same', activation='relu'),
            Conv2D(512, kernel_size=[3, 3], padding='same', activation='relu'),
            Conv2D(512, kernel_size=[3, 3], padding='same', activation='relu'),
            MaxPool2D(pool_size=[2, 2], strides=2, padding='same'),
            # fc
            Flatten(),
            Dense(4096, activation='relu'),
            Dense(4096, activation='relu'),
            Dense(10, activation='softmax')
        ]

        if self.bool_local_train:
            self.model = Sequential(vgg16_layers)
            self.model.build(input_shape=[None, 32, 32, 3])
            optimizer = optimizers.SGD(lr=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
            self.model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
            train_history = self.model.fit(x=self.x_train, y=self.y_train, epochs=20, batch_size=self.batch_size, verbose=2)
        else:
            self.model = km.load_model('./Hw1_5_model.h5')

    def func1(self):
        ax = []
        fig = plt.figure()
        for i in range(10):
            r = random.randint(0, 50000)
            ax.append(fig.add_subplot(2, 5, i+1))
            ax[-1].set_title(change_tag_to_string(self.trainY[r]))
            plt.imshow(self.x_train[r])
            plt.axis('off')
        fig.tight_layout()
        plt.show()

    def func2(self):
        print('hyperparameters:')
        print('batch size: ', self.batch_size)
        print('learning rate: ', self.learning_rate)
        print('optimizer: ' + 'SGD')

    def func3(self):
        print(self.model.summary())

    def func4(self):
        fig = plt.figure()
        fig.set_size_inches(12, 6)
        img1 = mpimg.imread('./accu.png')
        img2 = mpimg.imread('./loss.png')
        plt.subplot(1, 2, 1)
        plt.imshow(img1)
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(img2)
        plt.axis('off')
        plt.show()

    def func5(self):
        index = self.spinBox.value()
        img = np.array([self.x_test[index]])
        tag = []
        for i in range(10):
            tag.append(change_tag_to_string([i]))
        tag[0] = 'plane'
        tag[1] = 'car'
        pred = self.model.predict(img)
        #print(pred[0])

        fig = plt.figure()
        fig.set_size_inches(12, 6)
        plt.subplot(1, 2, 1)
        plt.imshow(img[0])
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.bar(tag, pred[0])
        plt.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    ui = MainUi()
    ui.show()
    sys.exit(app.exec_())

