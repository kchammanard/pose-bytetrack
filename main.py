import cv2
from ultralytics import YOLO
import time
import numpy as np
import cv2
from custom_socket import CustomSocket
import socket
import json
import numpy as np
import traceback
from tensorflow import keras
import tensorflow as tf
from keras.models import load_model


WEIGHT = "weights/yolov8s-pose.pt"
KERAS_WEIGHT = "first_weight"
DATASET_NAME = "coco"
# DATASET_NAME = {0: "coke"}
# DATASET_NAME = {0: "coke", 1: "milk", 2: "waterbottle"
# YOLOV8_CONFIG = {"tracker": "botsort.yaml",
#                  "conf": 0.7,
#                  "iou": 0.3,
#                  "show": True,
#                  "verbose": False}


YOLO_CONF = 0.7
KEYPOINTS_CONF = 0.7

def process_keypoints(keypoints, conf, frame_width, frame_height, origin = (0,0)):
    kpts = np.copy(keypoints)
    kpts[:,0] = (kpts[:,0] - origin[0]) / frame_width
    kpts[:,1] = (kpts[:,1] - origin[1]) / frame_height

    kpts[:,:-1][kpts[:,2] < conf] = [-1,-1]
    return np.round(kpts[:,:-1].flatten(),4)


def main():
    HOST = socket.gethostname()
    PORT = 12302

    server = CustomSocket(HOST, PORT)
    server.startServer()

    model = YOLO(WEIGHT, task="pose")

    keras_model = load_model(KERAS_WEIGHT)
    # keras_model = load_model("model/pose_estimation.h5",compile= False)

    while True:
        # Wait for connection from client :}
        conn, addr = server.sock.accept()
        print("Client connected from", addr)

        # start = time.time()

        # Process frame received from client
        while True:
            res = dict()
            try:
                data = server.recvMsg(conn, has_splitter=True)

                frame_height, frame_width = int(data[0]), int(data[1])
                # print(frame_height, frame_width)
                
                img = np.frombuffer(data[-1], dtype=np.uint8).reshape(frame_height, frame_width, 3)

                results = model.track(source=img, conf=YOLO_CONF, show=True, verbose=False, persist=True)[0]
                kpts = results.keypoints.cpu().numpy()
                boxes = results.boxes.data.cpu().numpy()

                for person_pred in zip(kpts, boxes):
                    person_kpts, person_box = person_pred
                    x1, y1, x2, y2 = person_box[:4]
                    person_id = int(person_box[4])

                    processed_kpts = process_keypoints(person_kpts, KEYPOINTS_CONF, frame_width, frame_height, (x1, y1))
                    # print(processed_kpts)

                    pred_pose = np.argmax(keras_model.predict(processed_kpts.reshape((1, 34)), verbose=0), axis=1)
                    print(pred_pose[0])

                    res[person_id] = int(pred_pose[0])

                    # Draw points
                    for i, pt in enumerate(person_kpts):
                        x, y, p = pt
                        if p >= KEYPOINTS_CONF:
                            cv2.putText(img, str(i), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Send back result
                # print(res)
                server.sendMsg(conn, json.dumps(res))

            except Exception as e:
                traceback.print_exc()
                print(e)
                print("Connection Closed")
                del res
                break

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
