import cv2
from ultralytics import YOLO

# Load your trained model
model = YOLO("best (1).pt")  # Change path if needed

# Initialize webcam
cap = cv2.VideoCapture(0)  # Use 0 or 1 based on your webcam index

# Get class names (ASL signs)
class_names = model.names  # e.g., {0: 'A', 1: 'B', ..., 25: 'Z'}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference on frame
    results = model.predict(source=frame, save=False, conf=0.5, imgsz=640, verbose=False)

    # Draw results
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label = f"{class_names[cls_id]} {conf:.2f}"

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("ASL Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
