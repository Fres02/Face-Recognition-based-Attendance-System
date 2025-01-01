import cv2

# Access the webcam
webcam_video = cv2.VideoCapture(0)

# Load the Haar cascade classifier
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = webcam_video.read()
    if not ret:
        print("Error: Unable to access the webcam.")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)

    # Mirror the frame horizontally
    frame = cv2.flip(frame, 1)

    cv2.imshow("Frame", frame)

    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Release the webcam and close windows
webcam_video.release()
cv2.destroyAllWindows()
