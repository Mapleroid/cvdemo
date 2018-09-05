import numpy as np
import cv2
 
def centroid_histogram(clt):
 
	numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
	(hist, _) = np.histogram(clt.labels_, bins = numLabels)
 
	hist = hist.astype("float")
	hist /= hist.sum()
 
	return hist
 
 
def plot_colors(hist, centroids):
 
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0
 
	for (percent, color) in zip(hist, centroids):
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
			color.astype("uint8").tolist(), -1)
		startX = endX
	
	return bar

def plot_colors2(main_colors):
 
	bar = np.zeros((50, 300, 3), dtype = "uint8")
	startX = 0
 
	for (color, percent) in main_colors:
		endX = startX + (percent * 300)
		cv2.rectangle(bar, (int(startX), 0), (int(endX), 50), color, -1)
		startX = endX
	
	bar = cv2.cvtColor(bar, cv2.COLOR_HSV2BGR)
	return bar