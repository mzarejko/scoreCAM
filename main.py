import numpy as np
import tensorflow as tf
import os
from cnn import create_premodel, scoreCAM, train
import tensorflow as tf
import matplotlib.pyplot as plt

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

SEED = 1000
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

model, cam = train(X_train, ohe_y_train, X_test, ohe_y_test)

#model = tf.keras.models.load_model('./model/model')
#cam = tf.keras.models.load_model('./model/cam')

camScore = scoreCAM(X_train[0], cam, model, ohe_y_train[0])
plt.imshow(X_train[0])
plt.imshow(camScore, alpha=0.8, cmap=plt.cm.gray)
plt.show()




