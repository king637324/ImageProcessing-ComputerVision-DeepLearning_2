from sys import int_info
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from time import sleep

import hw2Q5_ui as ui

import numpy as np
import tensorflow as tf
import cv2
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

batch_size = 32
image_size = (224, 224)

import sys
def random_erasing(img):
    p = 0.4 # erasing probability
    s_low, s_high = 0.02, 0.4
    r1, r2 = 0.03, 2
    p1 = np.random.uniform(0,1)
    if p1 < p:
        return img
    else:
        W, H = img.shape[1], img.shape[0]
        S = H * W
        while True:
            S_e = S * np.random.uniform(low=s_low, high=s_high)
            r_e = np.random.uniform(low=r1, high=r2)

            H_e = np.sqrt(S_e * r_e)
            W_e = np.sqrt(S_e / r_e)

            x_e = np.random.randint(0, W)
            y_e = np.random.randint(0, H)

            if x_e + W_e <= W and y_e + H_e <= H:
                img_erased = np.copy(img)
                img_erased[y_e:int(y_e + H_e + 1), x_e:int(x_e + W_e + 1), :] = np.random.uniform(0, 1)
                return img_erased


def Resnet50():
    input_shape = image_size + (3,)
    inputs = keras.Input(shape=input_shape)
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(inputs)
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='valid', kernel_initializer='he_normal', name='conv1')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    res_blocks = [3, 4, 6, 3]
    res_filters = [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]

    first_conv = 1
    for index, block in enumerate(res_blocks):  # 0, 3
        for layer in range(block):  # 3
            input_tensor = x
            for idx, f in enumerate(res_filters[index]):
                pad = 'valid'
                ksize = (1, 1)
                if idx > 0 and idx < 2:
                    ksize = (3, 3)
                    pad = 'same'

                strides = (1, 1)
                if first_conv == 1:
                    first_conv = 0

                elif idx == 0 and layer == 0:
                    strides = (2, 2)

                x = layers.Conv2D(f, ksize, strides=strides, kernel_initializer='he_normal', padding=pad)(x)
                #             print(block, layer, f, ksize, strides, pad)
                x = layers.BatchNormalization()(x)
                if idx < 2:
                    x = layers.Activation("relu")(x)

            if layer == 0:
                strides = (2, 2)
                if index == 0:
                    strides = (1, 1)

                shortcut = layers.Conv2D(res_filters[index][-1], (1, 1), strides=strides,
                                         kernel_initializer='he_normal')(input_tensor)
                shortcut = layers.BatchNormalization()(shortcut)
            else:
                #             print('i', ksize, strides)
                shortcut = input_tensor

            x = layers.add([x, shortcut])
            x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    outputs = x
    resnet50 = keras.Model(inputs, outputs)
    return resnet50


resnet50 = Resnet50()
resnet50.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
resnet50.load_weights('resnet_CatDog_earsing_20.h5')

train_datagen = ImageDataGenerator(
    rescale=1./255,
    preprocessing_function=random_erasing   # for cvdl random-earsing
    # zoom_range=0.5,
    # rescale=0.5
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_generator = val_datagen.flow_from_directory(
    'test/',
    class_mode='binary',
    target_size=image_size,
    batch_size=batch_size,
    shuffle=True, seed=76094614)



class Main(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.ButtonQ5_1.clicked.connect(self.Q5_1)
        self.ButtonQ5_2.clicked.connect(self.Q5_2)
        self.ButtonQ5_3.clicked.connect(self.Q5_3)
        self.ButtonQ5_4.clicked.connect(self.Q5_4)


    def Q5_1(self):
        print("-------------Q5_1-------------")

        resnet50.summary()
        
        print("-------------Q5_1 Finsh-------------\n")

    def Q5_2(self):
        print("-------------Q5_2-------------")

        cv2.imshow('TensorBoard', cv2.imread('TensorBoard_epoch_20.jpg'))
        
        print("-------------Q5_2 Finsh-------------\n")

    def Q5_3(self):
        print("-------------Q5_3-------------")

        # test_index = self.spinBox.value()

        label_names = ['Cat', 'Dog']
        picture, label = test_generator.next()
        predict = resnet50.predict(picture)

        # 印出預測貓狗的機率
        # print(predict[0])
        
        if predict[0] >= 0.5 :
            predict = 1  #預測為狗
        else :
            predict = 0  #預測為貓

        # cv2.namedWindow('視窗名稱',0) 0表示視窗大小可以改變
        cv2.namedWindow('Class:' + label_names[predict],0)
        cv2.imshow('Class:' + label_names[predict], picture[0])


        # label_names = ['Cat', 'Dog']
        # data, label = test_generator.next()
        # img = (data[0] * 255).astype('uint8')

        # plt.axis("off")
        # plt.title(label_names[int(label[0])])
        # plt.imshow(img)
        # plt.show()
        
        print("-------------Q5_3 Finsh-------------\n")
    
    def Q5_4(self):
        print("-------------Q5_4-------------")

        picture, label = test_generator.next()

        # 9張隨機的貓狗圖
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(random_erasing(picture[0]))
        
        plt.savefig('random_erasing.png')
        pic = cv2.imread('random_erasing.png')
        cv2.imshow('random_erasing', pic)

        pic = cv2.imread('Random-Erasing augmentation comparison.png')
        cv2.imshow('Random-Erasing augmentation comparison', pic)
        
        print("-------------Q5_4 Finsh-------------\n")
        

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())