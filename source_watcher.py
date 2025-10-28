import cv2
import numpy as np
import time
from datetime import datetime

# --- Load DNN person detector (MobileNet SSD) ---
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt",
                               "MobileNetSSD_deploy.caffemodel")

# Only need the "person" class from COCO labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

video = cv2.VideoCapture("rtsp://root:argoncube@localhost:10554/axis-media/media.amp")
person_present = False
print("\nStarting DNN-based AmBe source surveillance\n")


while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to capture frame.")
        break

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    detected = False

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        idx = int(detections[0, 0, i, 1])

        # only keep strong person detections
        if CLASSES[idx] == "person" and confidence > 0.5:
            detected = True
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Intruder! {confidence:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 255), 2)

    # --- Alert logic ---
    if detected and not person_present:
        person_present = True
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"⚠️ ALERT: PERSON DETECTED at {ts}")
        cv2.imwrite(f"alert_{int(time.time())}.jpg", frame)

    elif not detected and person_present:
        person_present = False
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Scene clear at {ts}")

    cv2.imshow("AmBe source surveillance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
