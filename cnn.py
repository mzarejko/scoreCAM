import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import time
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from tensorflow.keras.utils import plot_model
import cv2
from scikeras.wrappers import KerasClassifier
from skopt.space import Integer, Real
from skopt import BayesSearchCV

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

SEED = 1000
SHAPE_IMG = (32, 32, 3)
CONV_SHAPE = (-1, 2, 2, 1)
LABELS = 10

WIDTH_V3 = 128
HEIGHT_V3 = 128

np.random.seed(SEED)

N1 = Integer(124, 512)
N2 = Integer(124, 256)
DROP = Real(0.1, 0.5)
LR = Real(0.01, 0.1)
BATCH = 32
EPOCHS = 30

def create_premodel(n1=N1, n2=N2, epochs=EPOCHS, batch_size=BATCH, lr=LR, drop=DROP, paramsearch=True):
    ln = 'mixed10'

    v3 = InceptionV3(include_top=False, weights='imagenet', input_shape=(HEIGHT_V3, WIDTH_V3, 3))
    for layer in v3.layers:
        layer.treinable=False

    lconv = v3.get_layer(ln)

    CAMv3 = Model(v3.input, lconv.output)

    input4img = Input(shape=SHAPE_IMG)
    upimg = Resizing(HEIGHT_V3, WIDTH_V3)(input4img)
    v3img = CAMv3(upimg)

    v3mask = v3(upimg)
    fc = Flatten()(v3mask)
    d1 = Dense(n1, kernel_regularizer='l1')(fc)
    bn = BatchNormalization()(d1)
    act1 = Activation('relu')(bn)
    d2 = Dense(n2, kernel_regularizer='l1')(fc)
    bn2 = BatchNormalization()(d2)
    act1 = Activation('relu')(bn2)
    drop = Dropout(drop)(act1)
    out = Dense(10, activation='softmax')(drop)

    CAMnet = Model(input4img, v3img)
    trainNet = Model(input4img, out)

    trainNet.compile(optimizer=Adam(learning_rate=lr),
                    loss='categorical_crossentropy',
                    metrics=AUC())
    trainNet.summary()
    plot_model(trainNet, to_file="model.png", show_shapes=True)

    if paramsearch:
        return trainNet

    return CAMnet, trainNet


def train(X_train, y_train, X_test, y_test):

    wrapper = KerasClassifier(model = create_premodel,
                            drop=DROP,
                            n1=N2,
                            n2=N2,
                            lr=LR)


    bsc = BayesSearchCV(
        wrapper,
        {
            'n1': N1,
            'n2': N2,
            'drop': DROP,
            'lr': LR,
        },
        n_iter=32,
        random_state=SEED,
        cv=3,
        verbose=1,
        n_jobs=1
        )

    with tf.device('/gpu:0'):
        results = bsc.fit(X_train, y_train)

    print('n1 : ' + str(results.best_params_['n1']))
    print('n2 : ' + str(results.best_params_['n2']))
    print('dropout : ' + str(results.best_params_['drop']))
    print('lr : ' + str(results.best_params_['lr']))

    cam, good_model = create_premodel(results.best_params_['n1'],
                                 results.best_params_['n2'],
                                 results.best_params_['lr'],
                                 results.best_params_['drop'],
                                 paramsearch=False)

    early_stop = EarlyStopping(monitor='val_auc', patience=10)
    board = TensorBoard(log_dir=f'./logs/{time.time()}')
    checkpoints = ModelCheckpoint(filepath='./model/checkpoints',
                                  save_weights_only=True,
                                  monitor='val_auc',
                                  mode='max',
                                  save_best_only=True)

    with tf.device('/gpu:0'):
        good_model.fit(X_train, y_train,
                epochs=EPOCHS,
                validation_split=0.3,
                verbose=1,
                callbacks=[early_stop, board, checkpoints],
                shuffle=True,
                batch_size=BATCH)

    score = model.evaluate(X_test, y_test)
    print(f'score: {score}')
    good_model.save('./model/model')
    cam.save('./model/cam')
    return model, cam


def scoreCAM(img, CAMnet, trainNet, target):
    convs = CAMnet.predict(img[np.newaxis, :]).reshape(CONV_SHAPE)
    upconvs = []
    for c in convs:
        upc = cv2.resize(
            c, dsize=(SHAPE_IMG[0], SHAPE_IMG[1]), interpolation=cv2.INTER_AREA)
        upconvs.append(np.stack([upc, upc, upc], axis=-1))
    upconvs = np.array((upconvs))

    actmaps = np.multiply(upconvs, img)
    weights = []
    factmaps = []

    for m in actmaps:
        pred = trainNet.predict(m[np.newaxis, :])
        w = pred[:, np.argmax(target)][0]
        s = np.multiply(w, m)
        factmaps.append(s)

    print(np.array(factmaps).shape)

    camScore = np.sum(factmaps, axis=0)
    print(camScore.shape)
    for x in range(SHAPE_IMG[0]):
        for y in range(SHAPE_IMG[1]):
            camScore[x, y] = np.max(camScore[x, y], 0)
    return camScore
