import cv2
import os
import numpy as np

dataset_path = "dataset"

faces = []
labels = []
label_map = {}
current_id = 0

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_path):
        continue

    label_map[current_id] = person_name

    for image_name in os.listdir(person_path):
        img_path = os.path.join(person_path, image_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detections = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in detections:
            face = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
            faces.append(face)
            labels.append(current_id)

    current_id += 1

if len(faces) == 0:
    print("No faces found!")
    exit()

model = cv2.face.LBPHFaceRecognizer_create()
model.train(faces, np.array(labels))
model.save("model.yml")

np.save("labels.npy", label_map)

print("Training Done")