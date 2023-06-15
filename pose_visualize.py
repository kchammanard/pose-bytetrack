from ultralytics import YOLO
import cv2
import time
import numpy as np
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from ultralytics.yolo.utils.torch_utils import select_device
import yaml
from random import randint
import cv2
import numpy as np
import os


def list_available_cam(max_n):
    list_cam = []
    for n in range(max_n):
        cap = cv2.VideoCapture(n)
        ret, _ = cap.read()

        if ret:
            list_cam.append(n)
        cap.release()

    if len(list_cam) == 1:
        return list_cam[0]
    else:
        print(list_cam)
        return int(input("Cam index: "))


def draw_3d(frame, keypoints, joint_list):
    for joint in joint_list:
        cv2.line(frame, [int(k) for k in keypoints[joint[0]]], [
                 int(k) for k in keypoints[joint[1]]], (255, 0, 0), 2)


model = YOLO("weights/snack_pose.pt", task="pose")
cap = cv2.VideoCapture("data/snack_vid.mp4")

YOLO_CONF = 0.7
KEYPOINTS_CONF = 0.7

start = time.time()
FRAME_WIDTH = cap.get(3)
FRAME_HEIGHT = cap.get(4)

rand_color_list = np.random.rand(20, 3) * 255

while cap.isOpened():
    res = []
    ret, frame = cap.read()
    if not ret:
        print("Error")
        continue

    results = model.track(source=frame, conf=YOLO_CONF,
                          show=False, verbose=False, persist=True)[0]
    kpts = results.keypoints.cpu().numpy()
    boxes = results.boxes.data.cpu().numpy()
    # print(boxes)
    # print(kpts)

    for obj_kpts, obj_box in zip(kpts, boxes):
        # print(obj_box)
        x1, y1, x2, y2 = obj_box[:4]
        obj_id = int(obj_box[4])
        print(obj_id)

        cv2.rectangle(frame, (int(x1), int(y1)),
                      (int(x2), int(y2)), (0, 255, 0), 2)

        # Draw points
        for i, pt in enumerate(obj_kpts):
            x, y = pt
            cv2.putText(frame, str(i), (int(x), int(y)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        draw_3d(frame, obj_kpts, [(0, 1), (2, 3), (4, 5), (6, 7), (1, 2),
                (0, 3), (5, 6), (4, 7), (0, 7), (1, 6), (2, 5), (3, 4)])

    cv2.putText(frame, "fps: " + str(round(1 / (time.time() - start), 2)), (10, int(cap.get(4)) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    start = time.time()

    cv2.imshow("frame", frame)

    if cv2.waitKey(50) == ord("q"):
        cap.release()

cv2.destroyAllWindows()
