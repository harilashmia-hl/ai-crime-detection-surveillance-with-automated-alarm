import time
import cv2
from ultralytics import YOLO

try:
    import winsound  # Windows alarm
except ImportError:
    winsound = None

MODEL_PATH = "yolov8n.pt"            # replace with your custom crime model later
SUSPICIOUS = {"knife", "scissors"}   # classes that trigger the alarm
CONF = 0.5
FRAMES_NEEDED = 5                    # consecutive frames before alarm
COOLDOWN = 10                        # seconds between alarms


def alarm():
    if winsound:
        winsound.Beep(2000, 1000)
    else:
        print("\a ALARM!")


def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(0)        # 0 = webcam, or an RTSP/video path
    hits, last_alarm = 0, 0

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        result = model(frame, conf=CONF, verbose=False)[0]
        labels = {model.names[int(c)] for c in result.boxes.cls}
        hits = hits + 1 if labels & SUSPICIOUS else 0

        if hits >= FRAMES_NEEDED and time.time() - last_alarm > COOLDOWN:
            print(f"[ALERT] {labels & SUSPICIOUS}")
            alarm()
            last_alarm = time.time()
            hits = 0

        cv2.imshow("Surveillance", result.plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
