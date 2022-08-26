import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Multiply, Flatten, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

SEED = 1000
SHAPE_IMG = (32, 32, 3)
LABELS = 10

WIDTH_V3 = 128
HEIGHT_V3 = 128

CONV_SHAPE = (2, 2, 1)
BATCH = 32
LR = 0.01
EPOCHS = 30

np.random.seed(SEED)

from tensorflow.keras.datasets import cifar10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255
print(f"X: {X_train.shape}, y: {y_train.shape}")

def convert_to_OHE(data):
    unq = np.unique(data)
    ohe=[]
    for s in data:
        val = np.where(unq == s, 1, 0)
        ohe.append(val)
    ohe = np.array(ohe)
    return ohe

ohe_y_train = convert_to_OHE(y_train)
ohe_y_test = convert_to_OHE(y_test)

for y in range(len(y_test)):
    assert np.argmax(ohe_y_test[y]) == y_test[y]

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import time
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.utils import plot_model

def create_premodel(lname='mixed10'):

    v3 = InceptionV3(include_top=False, weights='imagenet', input_shape=(HEIGHT_V3, WIDTH_V3, 3))
    for layer in v3.layers:
        layer.treinable=False

    lconv = v3.get_layer(lname)

    CAMv3 = Model(v3.input, lconv.output)

    input4img = Input(shape=SHAPE_IMG)
    upimg = Resizing(HEIGHT_V3, WIDTH_V3)(input4img)
    v31img = CAMv3(upimg)

    convNet = Model(input4img, v31img)

    input4conv = Input(shape=CONV_SHAPE)
    upconv32 = Resizing(SHAPE_IMG[0], SHAPE_IMG[1])(input4conv)
    actmap32 = Activation('sigmoid')(upconv32)

    upconv128 = Resizing(HEIGHT_V3, WIDTH_V3)(input4conv)

    actmapNorm = Activation('sigmoid')(upconv128)
    mask = Multiply()([actmapNorm, upimg])

    v3mask = v3(mask)
    fc = Flatten()(v3mask)
    d1 = Dense(512, kernel_regularizer='l1')(fc)
    bn = BatchNormalization()(d1)
    act1 = Activation('relu')(bn)
    d2 = Dense(256, kernel_regularizer='l1')(fc)
    bn2 = BatchNormalization()(d2)
    act1 = Activation('relu')(bn2)
    drop = Dropout(0.4)(act1)
    out = Dense(10, activation='softmax')(drop)

    CAMnet = Model([input4img, input4conv], [out, actmap32, v31img])
    trainNet = Model([input4img, input4conv], out)

    trainNet.compile(optimizer=Adam(learning_rate=LR),
                    loss=['categorical_crossentropy', None, None],
                    metrics=AUC())
    CAMnet.summary()
    plot_model(CAMnet, to_file="model.png", show_shapes=True)
    return CAMnet, trainNet


def train(model, X_train, y_train, X_test, y_test):
    early_stop = EarlyStopping(monitor='val_auc', patience=10)
    board = TensorBoard(log_dir=f'./logs/{time.time()}')
    checkpoints = ModelCheckpoint(filepath='./model/checkpoints',
                                  save_weights_only=True,
                                  monitor='val_auc',
                                  mode='max',
                                  save_best_only=True)

    #add img of ones for input4conv in net while train to get same resulat after Multiply()
    unaffecting_train = []
    unaffecting_test = []
    for _ in range(len(X_train)):
        unaffecting_train.append(np.ones(CONV_SHAPE))
    unaffecting_train = np.array(unaffecting_train)


    for _ in range(len(X_test)):
        unaffecting_test.append(np.ones(CONV_SHAPE))
    unaffecting_test = np.array(unaffecting_test)

    with tf.device('/gpu:0'):
        model.fit([X_train, unaffecting_train], [y_train],
                epochs=EPOCHS,
                validation_split=0.3,
                verbose=1,
                callbacks=[early_stop, board, checkpoints],
                shuffle=True,
                batch_size=BATCH)

    score = model.evaluate([X_test, unaffecting_test], y_test)
    print(f'score: {score}')
    model.save('./model/ultimate_model')
    return score

def scoreCAM(img, CAMnet, convnet, net, target):
    maps = convnet.predict(img[np.newaxis, :])
    maps = maps.reshape([-1, 2, 2])
    weights = []
    factmaps = []

    for m in maps:
        actmap = CAMnet.predict([img[np.newaxis, :], m[np.newaxis, :]])
        actmap3 = np.stack([actmap, actmap, actmap]).reshape(1, 32, 32, 3)
        pred = net.predict([actmap3,
                            np.ones(CONV_SHAPE)[np.newaxis, :]])
        w = pred[:, np.argmax(target)][0]
        factmaps.append(np.multiply(w, actmap[0]))
    print(np.array(factmaps).shape)

    camScore = np.sum(factmaps, axis=0)
    print(camScore.shape)
    for x in range(SHAPE_IMG[0]):
        for y in range(SHAPE_IMG[1]):
            camScore[x, y] = np.max(camScore[x, y], 0)
    return camScore



CAMnet, net = create_premodel()
train(net, X_train, ohe_y_train,
        X_test, ohe_y_test)

convNet.save('./model/conv_model')
CAMnet.save('./model/cam_model')
trainNet.save('./model/train_net')

camScore = scoreCAM(X_train[0], CAMnet, convNet, trainNet, ohe_y_train[0])
plt.imshow(X_train[0])
plt.imshow(camScore, alpha=0.5)
plt.show()




