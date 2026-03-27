import cv2
import os 
import numpy as np
import pandas as pd
from datetime import datetime


# DATASET FOLDER

dataset_path = "dataset"

if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# FACE DETECTOR
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

recognizer = cv2.face.LBPHFaceRecognizer_create()

labels = {}
faces = []
ids = []

current_id = 0

# LOAD DATASET

for person in os.listdir(dataset_path):

    labels[current_id] = person
    person_path = f"{dataset_path}/{person}"

    for img in os.listdir(person_path):

        img_path = f"{person_path}/{img}"
        gray = cv2.imread(img_path, 0)

        if gray is not None:
            faces.append(gray)
            ids.append(current_id)

    current_id += 1

# Train model if dataset exists
if len(faces) > 0:
    recognizer.train(faces, np.array(ids))


# ATTENDANCE FUNCTION
def markAttendance(present_names):

    file = "Attendance.csv"

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    all_students = list(labels.values())

    rows = []

    for student in all_students:

        status = "Present" if student in present_names else "Absent"

        rows.append([student, date, time, status])

    df = pd.DataFrame(rows, columns=["Name", "Date", "Time", "Status"])

    df.to_csv(file, index=False)

    print("Attendance saved")

# CAMERA
cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)

print("Press N to register new face")
print("Press Q to quit and save attendance")

present_today = set()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Camera error")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces_detected = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces_detected:

        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))

        label = "Unknown"

        if len(faces) > 0:

            id_, conf = recognizer.predict(face_img)

            if conf < 80:

                label = labels[id_]
                present_today.add(label)

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Attendance System", frame)

    key = cv2.waitKey(10)

    # Register new face
    if key == ord('n'):

        name = input("Enter name: ")

        save_dir = f"{dataset_path}/{name}"

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        cv2.imwrite(f"{save_dir}/{len(os.listdir(save_dir))+1}.jpg", face_img)

        print("Face saved. Restart program to train.")

    # Quit and save attendance
    if key == ord('q'):
        markAttendance(present_today)
        break

cap.release()
cv2.destroyAllWindows()

