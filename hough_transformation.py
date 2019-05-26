import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('D:/Jannes/Dokumente/Uni/introCV/00000_00009.ppm', -1)
img = cv2.GaussianBlur(img, (3,3), 0)
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#cv2.imshow('image', img)
#cv2.waitKey()

#get edges
laplacian = cv2.Laplacian(img,cv2.CV_64F)

cv2.imshow('laplace', laplacian)
cv2.waitKey()

#forward laplacian
#image might need more blurr or will be improved through ROI