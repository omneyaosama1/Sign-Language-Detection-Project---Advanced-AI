import cv2
from ultralytics import YOLO

model = YOLO("../models/letters_model.pt")
cap = cv2.VideoCapture(0)
class_names = model.names

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, save=False, conf=0.5, imgsz=640, verbose=False)

    predictions = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = class_names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if (x2 - x1) < 30 or (y2 - y1) < 30:
                continue

            predictions.append((conf, label, (x1, y1, x2, y2)))

    predictions.sort(reverse=True, key=lambda x: x[0])

    for idx, (conf, label, (x1, y1, x2, y2)) in enumerate(predictions):
        color = (0, 255, 0) if idx == 0 else (0, 150, 150)  # Highlight top prediction
        text = f"{label} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("ASL Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
