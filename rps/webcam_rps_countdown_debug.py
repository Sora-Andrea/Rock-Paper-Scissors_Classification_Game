import time
import cv2
from ultralytics import YOLO

MODEL_PATH = "rps_yolo11_cls/yolo11x_rps/weights/best.pt"
ROI_SIZE = 280

def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    last_preds = None

    print("Controls:")
    print("  SPACE = 3-second countdown + snap prediction")
    print("  ESC   = quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        x1 = (w - ROI_SIZE) // 2
        y1 = (h - ROI_SIZE) // 2
        x2 = x1 + ROI_SIZE
        y2 = y1 + ROI_SIZE

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "SPACE: countdown+snap | ESC: quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if last_preds is not None:
            preds_text = ", ".join(f"{label} ({conf:.2f})" for label, conf in last_preds)
            cv2.putText(frame, f"Last: {preds_text}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("RPS Snapshot (YOLO11-CLS)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        if key == 32:  # SPACE
            # Countdown overlay
            for t in [3, 2, 1]:
                ret2, f2 = cap.read()
                if not ret2:
                    break
                cv2.rectangle(f2, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(f2, f"Capturing in {t}...", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.imshow("RPS Snapshot (YOLO11-CLS)", f2)
                cv2.waitKey(1)
                time.sleep(1)

            # Snapshot frame
            ret3, snap = cap.read()
            if not ret3:
                continue

            roi = snap[y1:y2, x1:x2]

            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results = model(roi_rgb, verbose=False)[0]
            probs = results.probs

            if probs is not None:
                scores = probs.data.tolist()
                ordered = sorted(
                    ((results.names[idx], float(score)) for idx, score in enumerate(scores)),
                    key=lambda item: item[1],
                    reverse=True,
                )
                last_preds = ordered[:3]
            else:
                last_preds = [("unknown", 0.0)]

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
