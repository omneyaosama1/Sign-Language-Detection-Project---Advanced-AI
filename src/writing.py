import cv2
from ultralytics import YOLO

model = YOLO("../models/letters_model.pt")
cap = cv2.VideoCapture(0)
class_names = model.names

confirmed_text = ""
last_letter = ""
consecutive_frames = 0
required_frames = 5 

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, save=False, conf=0.5, imgsz=640, verbose=False)

    detected_letter = None
    for r in results:
        boxes = r.boxes
        if len(boxes) > 0:
            top_box = max(boxes, key=lambda box: float(box.conf[0]))
            conf = float(top_box.conf[0])
            if conf > 0.5:
                cls_id = int(top_box.cls[0])
                detected_letter = class_names[cls_id]

                x1, y1, x2, y2 = map(int, top_box.xyxy[0])
                label = f"{detected_letter} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if detected_letter == last_letter and detected_letter is not None:
        consecutive_frames += 1
    else:
        consecutive_frames = 1
        last_letter = detected_letter

    if consecutive_frames >= required_frames and detected_letter:
        confirmed_text += detected_letter
        print(f"Confirmed: {confirmed_text}")
        consecutive_frames = 0
        last_letter = None 

    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        confirmed_text += ' '
    elif key == 8: 
        confirmed_text = confirmed_text[:-1]
    elif key == ord('q'):
        break

    cv2.putText(frame, f"Text: {confirmed_text}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("Spell ASL Letters", frame)

cap.release()
cv2.destroyAllWindows()
