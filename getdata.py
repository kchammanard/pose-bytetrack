from ultralytics import YOLO
import cv2
import time
import numpy as np
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from ultralytics.yolo.utils.torch_utils import select_device
import yaml
from random import randint
from ultralytics.SORT import *
import pandas as pd
import os
import csv

model = YOLO("yolov8s-pose.pt")

l=[]
for c in range(10):
    try:
        ret, frame = cv2.VideoCapture(c).read()

        if ret: 
            l.append(c)
    except:
        pass
print(l)

start = time.time()
cap = cv2.VideoCapture("data/3.mp4")

rand_color_list = np.random.rand(20, 3) * 255

def save_data_to_csv(data, label, filename):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        flattened_data = data.flatten()
        row = np.concatenate((flattened_data, [label]))
        writer.writerow(row)

while cap.isOpened():
    res = []
    ret, frame = cap.read()
    if not ret:
        print("Error")
        continue

    cv2.putText(frame, "fps: " + str(round(1 / (time.time() - start), 2)), (10, int(cap.get(4)) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    start = time.time()

    results = model.predict(source=frame, conf=0.7, show = True)[0]
  
    print(results.boxes)
    print(results.keypoints)
    print(results.keypoints.shape)

    if results.boxes:
        label = 3
        save_data_to_csv(results.keypoints,label,"data/test.csv")

    if cv2.waitKey(1) == ord("q"):
        cap.release()

cv2.destroyAllWindows()
