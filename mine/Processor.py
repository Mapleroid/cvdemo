import cv2
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import ctypes
import win32gui, win32ui, win32con, win32api

import utils

def get_kmeans_color(img, count, x, y):
    #t0 = time.time()
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    clt = KMeans(n_clusters = count)
    clt.fit(img)

    hist = utils.centroid_histogram(clt)
    colors = clt.cluster_centers_.astype("uint8").tolist()

    if hist[0] > hist[1]:
        return [(hist[0], colors[0]),
                (hist[1], colors[1])]
    else:
        return [(hist[1], colors[1]),
                (hist[0], colors[0])]