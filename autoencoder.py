from tensorflow.keras.layers import Dense, LeakyReLU, Flatten, BatchNormalization, SeparableConv2D, Conv2D, MaxPooling2D, Input, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow.keras import Model
import os


class Autoencoder:
    @classmethod
    def create_auto(shape):
        # encoder
        aut_in = Input(shape=shape)

        conv1 = Conv2D(32, (1, 1), padding='same')(aut_in)
        bn1 = BatchNormalization()(conv1)
        act1 = Activation(LeakyReLU())(bn1)

        conv2 = SeparableConv2D(64, (4, 4), padding='same')(act1)
        bn2 = BatchNormalization()(conv2)
        act2 = Activation(LeakyReLU())(bn2)
        pool1 = MaxPooling2D((2, 2), padding='same')(act2)

        conv3 = SeparableConv2D(128, (3, 3), padding='same')(pool1)
        bn3 = BatchNormalization()(conv3)
        act3 = Activation(LeakyReLU())(bn3)
        pool2 = MaxPooling2D((2, 2), padding='same')(act3)

        conv4 = SeparableConv2D(256, (3, 3), padding='same')(pool2)
        bn4 = BatchNormalization()(conv4)
        act4 = Activation(LeakyReLU())(bn4)
        pool3 = MaxPooling2D((2, 2), padding='same')(act4)
        out_1 = Flatten()(pool3)

        # decoder
        aut_d1 = Dense(512, activation = LeakyReLU())(out_1)
        aut_d2 = Dense(1024, activation=LeakyReLU())(aut_d1)
        aut_d3 = Dense(2048, activation=LeakyReLU())(aut_d2)
        out = Dense(784, activation="sigmoid")(aut_d3)

        model = Model(aut_in, out)
        model.compile(optimizer=Adam(learning_rate=0.005), loss='binary_crossentropy', metrics=AUC())
        print(model.summary())
        return model 

    @classmethod
    def train_auto(model, X_train, X_test, shape):
        early_stop = EarlyStopping(monitor='val_loss', patience=10)

        labels = X_train.reshape(len(X_train), shape[0]*shape[1])
        test_labels = X_train.reshape(len(X_test), shape[0]*shape[1])
        with tf.device('/gpu:0'):
            model.fit(X_train,  labels,
                    epochs=100,   
                    validation_data=(X_test, test_labels), 
                    verbose=2, 
                    callbacks=[early_stop], 
                    shuffle=True,
                    batch_size=128)

        model.save('./model')