from ultralytics import YOLO
import cv2

# Load yolo YOLOv8 model
model = YOLO("yolov8n.pt")

# Open webcam 
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detection
    results = model(frame, stream=True)

    for r in results:
        annotated_frame = r.plot()

    # Display
    cv2.imshow("YOLOv8 Webcam Detection", annotated_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()