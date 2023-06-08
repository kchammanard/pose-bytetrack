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


WEIGHT = "weights/yolov8s-pose.pt"
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

def process_keypoints(keypoints, conf, frame_width, frame_height):
    kpts = np.copy(keypoints)
    kpts[:,0] = kpts[:,0] / frame_width
    kpts[:,1] = kpts[:,1] / frame_height

    kpts[:,:-1][kpts[:,2] < conf] = [-1,-1]
    return kpts[:,:-1].flatten()


def main():
    HOST = socket.gethostname()
    PORT = 13000

    server = CustomSocket(HOST, PORT)
    server.startServer()

    model = YOLO(WEIGHT)

    keras_model = ""
    # keras_model = load_model("model/pose_estimation.h5",compile= False)

    while True:
        # Wait for connection from client :}
        conn, addr = server.sock.accept()
        print("Client connected from", addr)

        start = time.time()

        # Process frame received from client
        while True:
            res = dict()
            try:
                data = server.recvMsg(conn, has_splitter=True)

                frame_height, frame_width = int(data[0]), int(data[1])
                # print(frame_height, frame_width)
                
                img = np.frombuffer(data[-1], dtype=np.uint8).reshape(frame_height, frame_width, 3)

                results = model.predict(source=img, conf=YOLO_CONF, show=True, verbose=False)[0]
                kpts = results.keypoints.cpu().numpy()

                for id, person_kpts in enumerate(kpts):

                    processed_kpts = process_keypoints(person_kpts, KEYPOINTS_CONF, frame_height, frame_width)
                    print(processed_kpts)

                    # pred_pose = np.argmax(keras_model.predict(processed_kpts, verbose=0), axis=1)
                    pred_pose = 1

                    res[id] = pred_pose

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