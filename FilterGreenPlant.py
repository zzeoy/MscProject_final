import numpy as np
import cv2 as cv
import pandas as pd
import csv
filename='GB.png'
Test=cv.imread(filename)
TestHSV=cv.cvtColor(Test,cv.COLOR_BGR2HSV)
mask = cv.inRange(TestHSV, (10, 10, 40), (120, 255,255))
imask = mask>0
green = np.zeros_like(Test, np.uint8)
green[imask] = Test[imask]
## save
r,g,b=cv.split(green)
gh=cv.equalizeHist(g)
cv.imwrite(filename, g)
