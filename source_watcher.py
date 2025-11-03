import cv2
import numpy as np
import time
from datetime import datetime
import threading
import tk


# -----------------------------------------------------------
# Helper function to build the RTSP URL
# -----------------------------------------------------------
def build_camera_url(local_mode, user, password, ip, forward_port=None):
    if local_mode:
        return f"rtsp://{user}:{password}@{ip}/axis-media/media.amp"
    else:
        if forward_port is None:
            raise ValueError("Forward port must be provided for tunnel mode.")
        return f"rtsp://{user}:{password}@localhost:{forward_port}/axis-media/media.amp"


# -----------------------------------------------------------
# Popup alert window (tkinter, persistent until closed)
# -----------------------------------------------------------
def show_popup_alert(message):
    """
    Display a popup alert window always on top of other windows.
    Stays open until the user closes it manually.
    """
    def _popup():
        root = tk.Tk()
        root.title("⚠️ Intruder Alert!")
        root.attributes('-topmost', True)       # keep window on top
        root.geometry("420x160+600+350")        # size and position
        root.configure(bg='red')

        label = tk.Label(root,
                         text=message,
                         font=("Arial", 16, "bold"),
                         fg="white",
                         bg="red",
                         wraplength=380,
                         justify="center")
        label.pack(expand=True, fill="both", padx=20, pady=(20, 10))

        btn = tk.Button(root,
                        text="Dismiss",
                        font=("Arial", 14, "bold"),
                        fg="white",
                        bg="black",
                        command=root.destroy)
        btn.pack(pady=(0, 20))

        root.mainloop()

    threading.Thread(target=_popup, daemon=True).start()


# -----------------------------------------------------------
# Main DNN surveillance function
# -----------------------------------------------------------
def run_surveillance(camera_url):
    net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt",
                                   "MobileNetSSD_deploy.caffemodel")

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    video = cv2.VideoCapture(camera_url)
    if not video.isOpened():
        print(f"❌ Failed to open camera stream at {camera_url}")
        return

    print("\nStarting DNN-based AmBe source surveillance\n")

    person_present = False
    last_alert_time = 0
    alert_cooldown = 300  # 5 minutes

    try:
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

                if CLASSES[idx] == "person" and confidence > 0.5:
                    detected = True
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"Intruder! {confidence:.2f}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 2)

            current_time = time.time()

            if detected and not person_present:
                person_present = True
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if current_time - last_alert_time > alert_cooldown:
                    print(f"🚨 ALERT: PERSON DETECTED at {ts}")
                    cv2.imwrite(f"alert_{int(current_time)}.jpg", frame)
                    last_alert_time = current_time

                    # Persistent popup alert
                    show_popup_alert(f"Person detected at {ts}\n\n(Press 'Dismiss' to close)")
                else:
                    remaining = int(alert_cooldown - (current_time - last_alert_time))
                    print(f"Person detected but still in cooldown ({remaining}s left)")

            elif not detected and person_present:
                person_present = False
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"Scene clear at {ts}")

            cv2.imshow("AmBe source surveillance", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exit requested via keyboard.")
                break

            if cv2.getWindowProperty("AmBe source surveillance", cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed, exiting surveillance.")
                break

    except KeyboardInterrupt:
        print("\n Program interrupted by user.")

    video.release()
    cv2.destroyAllWindows()


# -----------------------------------------------------------
# Main execution
# -----------------------------------------------------------
if __name__ == "__main__":
    print("=== AmBe Source Surveillance Configuration ===")
    local_or_tunnel = input("Run locally or via tunnel? [local/tunnel]: ").strip().lower()
    user = input("Camera username: ").strip()
    password = input("Camera password: ").strip()
    ip = input("Camera IP address: ").strip()

    if local_or_tunnel == "local":
        camera_url = build_camera_url(True, user, password, ip)
    elif local_or_tunnel == "tunnel":
        forward_port = input("Forwarded local port (e.g., 10443): ").strip()
        camera_url = build_camera_url(False, user, password, ip, forward_port)
    else:
        raise ValueError("Invalid mode. Choose 'local' or 'tunnel'.")

    print(f"\nConnecting to camera stream: {camera_url}\n")
    print("Press 'q' in the window, close the window, or type Ctrl+C to exit.\n")

    run_surveillance(camera_url)
