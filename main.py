import os
import re
import time
import cv2
import numpy as np
from os.path import isfile, join
import matplotlib.pyplot as plt

tree_classifier = cv2.CascadeClassifier('/Users/christianekeigwe/Desktop/Desktop/Purdue/Summer 2021/CNIT 39000/treedetection/cascade.xml')

cap = cv2.VideoCapture('/Users/christianekeigwe/Desktop/Desktop/Purdue/Summer 2021/CNIT 39000/treedetection/DJI_0014.MP4')

#"""
tree_classifier = cv2.CascadeClassifier('/Users/christianekeigwe/Desktop/Desktop/Purdue/Summer 2021/CNIT 39000/treedetection/cascade.xml')

cap = cv2.VideoCapture('/Users/christianekeigwe/Desktop/Desktop/Purdue/Summer 2021/CNIT 39000/treedetection/DJI_0014.MP4')

while 1:
    time.sleep(.05)
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    trees = tree_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in trees:
        image = cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)

    cv2.imshow('Trees', image)
    cv2.waitKey()

cap.release()
#cv2.destroyAllWindows()
#"""
