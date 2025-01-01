from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(str1):
    speak = Dispatch("SAPI.SpVoice")
    speak.Speak(str1)

webcam_video = cv2.VideoCapture(0)
if not webcam_video.isOpened():
    print("Error: Webcam not initialized.")
    exit()

cascade_path = 'data/haarcascade_frontalface_default.xml'
facedetect = cv2.CascadeClassifier(cascade_path)
if facedetect.empty():
    print(f"Error: Could not load Haar cascade from {cascade_path}. Check the file path.")
    exit()

try:
    with open('data/names.pkl', 'rb') as w:
        LABELS = pickle.load(w)
    with open('data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

print('Shape of Faces matrix --> ', FACES.shape)
if len(FACES.shape) == 4:  # If it's 4D, reshape to 2D
    FACES = FACES.reshape(FACES.shape[0], -1)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

background_path = "background.png"
imgBackground = cv2.imread(background_path)
if imgBackground is None:
    print(f"Error: Could not load background image from {background_path}. Check the file path.")
    exit()

FRAME_WIDTH = 480 
FRAME_HEIGHT = 360  

BACKGROUND_HEIGHT, BACKGROUND_WIDTH, _ = imgBackground.shape
start_y = (BACKGROUND_HEIGHT - FRAME_HEIGHT) // 2
start_x = (BACKGROUND_WIDTH - FRAME_WIDTH) // 2

COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = webcam_video.read()
    if not ret:
        print("Failed to capture frame from the webcam.")
        continue

    resized_frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    attendance = []

    for (x, y, w, h) in faces:
        crop_img = resized_frame[y:y + h, x:x + w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")

        cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.rectangle(resized_frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(resized_frame, str(output[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        attendance = [str(output[0]), str(timestamp)]

    try:
        imgBackground_copy = imgBackground.copy() 
        imgBackground_copy[start_y:start_y + FRAME_HEIGHT, start_x:start_x + FRAME_WIDTH] = resized_frame
    except ValueError as e:
        print(f"Error: Could not overlay resized frame on the background. {e}")
        continue

    cv2.imshow("Frame", imgBackground_copy)

    k = cv2.waitKey(1)
    if k == ord('o'):
        speak("Attendance Taken.")
        time.sleep(1)
        if exist:
            with open("Attendance/Attendance_" + date + ".csv", "a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
        else:
            with open("Attendance/Attendance_" + date + ".csv", "a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)

    if k == ord('q'):
        break

webcam_video.release()
cv2.destroyAllWindows()
