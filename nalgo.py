import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
from tensorflow.keras.models import load_model

SEED = 1000
WIDTH = 28
HEIGHT = 28

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

(X_train, y_train), (X_test, y_test) = mnist.load_data()


# test quality of data
assert np.isnan(X_train).sum() == 0
assert np.isnan(y_train).sum() == 0
print(f"X: {X_train.shape}, y: {y_train.shape}")

# scale
X_train = X_train/255

# check outliers

def rm_outliers_LOF(img):
    pass

def rm_outliers_auto(img):
    img = img.reshape(1, 28, 28, 1)
    model = load_model('./model')
    img_out = model.predict(img)
    img_out = img_out.reshape(28, 28)
    return img_out



def rm_outliers_Iforest(img, threshold=0.9):
    assert type(threshold) == float

    img_outliers = copy.deepcopy(img).reshape(-1,1)
    iforests = IsolationForest(n_estimators=10, random_state=SEED)
    iforests.fit(img_outliers)
    scores = -1*iforests.score_samples(img_outliers)
    scores = scores.reshape(WIDTH, HEIGHT)
    img_outliers = img_outliers.reshape(WIDTH, HEIGHT)
    img_rm_outliers = np.where(scores > threshold, 0, img_outliers)

    return scores, img_rm_outliers

def diagnostic_plot(img1, img3, sc=None, i=0):
    if sc:
        fig, ax = plt.subplots(1, 3, figsize=(16,7))

        ax[0].imshow(img1)
        ax[0].set_title("Image with outliers")

        omi = ax[1].imshow(sc, cmap='RdBu')
        ax[1].set_title("Isolation Forest score")
        fig.colorbar(omi, ax=ax[1], label="Outliers")

        ax[2].imshow(img3)
        ax[2].set_title("Image without outliers")
    else:
        fig, ax = plt.subplots(1, 2, figsize=(16,7))
        ax[0].imshow(img1)
        ax[0].set_title("Image with outliers")
        ax[1].imshow(img3)
        ax[1].set_title("Image without outliers")

    plt.savefig(f'./iforest/Iforest{i}.jpg')
    plt.close()

img1 = rm_outliers_auto(X_train[0])
diagnostic_plot(X_train[0], img1)




