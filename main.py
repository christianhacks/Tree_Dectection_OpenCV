import os
import re
import time
import cv2
import numpy as np
from os.path import isfile, join

tree_classifier = cv2.CascadeClassifier('<Cascade_File_Path>')

cap = cv2.VideoCapture('<Video_File_Path>')

while True:
    time.sleep(.05)
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    trees = tree_classifier.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in trees:
        image = cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)
        cv2.imshow('Trees', image)
        #cv2.namedWindow('Trees', cv2.WINDOW_NORMAL) #optional
        #cv2.resizeWindow('Trees', 1900, 1000) #optional
        cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
