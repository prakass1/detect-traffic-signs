import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('D:/Jannes/Dokumente/Uni/introCV/00000_00009.ppm',-1)
#cv2.imshow('image', img)
#cv2.waitKey()
#get red, blue, green histograms
#r,g,b = cv2.split(img)
r_hist = cv2.calcHist([img],[0], None, [256],[0,256])
g_hist = cv2.calcHist([img],[1], None, [256],[0,256])
b_hist = cv2.calcHist([img],[2], None, [256],[0,256])
#plt.hist(img.ravel(),256,[0,256]); plt.show()
'''
#other way:
#from https://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()
'''