import cv2
import numpy as np
import time
import shutil
import subprocess
import os
from datetime import datetime

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
# Terminal-based alert (all red text)
# -----------------------------------------------------------
def show_terminal_alert(message):
    """
    Open a new terminal window displaying an alert message in red.
    The terminal remains open until the user closes it.
    Works with common Linux terminal emulators.
    """
    RED = "\033[31m"
    RESET = "\033[0m"

    # Entire message in red
    alert_text = f"{RED}INTRUDER ALERT\n\n{message}\n\nThis terminal will remain open until you close it.{RESET}"

    # Detect a terminal emulator
    terminal = shutil.which("xterm") or shutil.which("gnome-terminal") \
               or shutil.which("konsole") or shutil.which("xfce4-terminal") \
               or shutil.which("terminator") or shutil.which("urxvt")

    if not terminal:
        print("No terminal emulator found. Printing alert to console:")
        print(alert_text)
        return

    bash_command = "cat <<'EOF'\n" + alert_text + "\nEOF\nexec bash"

    try:
        tname = terminal.split("/")[-1]
        if tname == "xterm" or tname == "urxvt":
            cmd = [terminal, "-hold", "-geometry", "80x12+600+300", "-T", "Intruder Alert",
                   "-e", "bash", "-lc", bash_command]
        elif tname == "gnome-terminal":
            cmd = [terminal, "--window", "--", "bash", "-lc", bash_command]
        elif tname == "xfce4-terminal":
            cmd = [terminal, "--hold", "--geometry=80x12+600+300", "--title=Intruder Alert",
                   "--", "bash", "-lc", bash_command]
        elif tname == "konsole":
            cmd = [terminal, "--hold", "--new-tab", "-p", "tabtitle=Intruder Alert",
                   "-e", "bash", "-lc", bash_command]
        elif tname == "terminator":
            cmd = [terminal, "-x", "bash", "-lc", bash_command]
        else:
            cmd = [terminal, "-e", "bash", "-lc", bash_command]

        subprocess.Popen(cmd)
    except Exception as e:
        print("Failed to open terminal for alert:", e)
        print(alert_text)


# -----------------------------------------------------------
# Main DNN surveillance function
# -----------------------------------------------------------
def run_surveillance(camera_url):
    """
    Run DNN-based person detection on an RTSP camera stream.
    """
    net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt",
                                   "MobileNetSSD_deploy.caffemodel")

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    # Create directory for alert images
    alert_dir = "alerts"
    if not os.path.exists(alert_dir):
        os.makedirs(alert_dir)
        print(f"Created directory '{alert_dir}' for alert images.")
    else:
        print(f"Using existing directory '{alert_dir}' for alert images.")

    video = cv2.VideoCapture(camera_url)
    if not video.isOpened():
        print(f"Failed to open camera stream at {camera_url}")
        return

    print("\nStarting DNN-based surveillance\n")

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
                confidence = float(detections[0, 0, i, 2])
                idx = int(detections[0, 0, i, 1])

                if CLASSES[idx] == "person" and confidence > 0.5:
                    detected = True
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"Person {confidence:.2f}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (0, 0, 255), 2)

            current_time = time.time()

            if detected and not person_present:
                person_present = True
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if current_time - last_alert_time > alert_cooldown:
                    print(f"ALERT: PERSON DETECTED at {ts}")
                    filename = os.path.join(alert_dir, f"alert_{int(current_time)}.jpg")
                    cv2.imwrite(filename, frame)
                    last_alert_time = current_time

                    show_terminal_alert(f"Person detected at {ts}\nImage saved: {filename}")
                else:
                    remaining = int(alert_cooldown - (current_time - last_alert_time))
                    print(f"Person detected but still in cooldown ({remaining}s left)")

            elif not detected and person_present:
                person_present = False
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"Scene clear at {ts}")

            cv2.imshow("Surveillance", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exit requested via keyboard.")
                break

            if cv2.getWindowProperty("Surveillance", cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed, exiting surveillance.")
                break

    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")

    video.release()
    cv2.destroyAllWindows()


# -----------------------------------------------------------
# Main execution
# -----------------------------------------------------------
if __name__ == "__main__":
    print("=== Surveillance Configuration ===")
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
