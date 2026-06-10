import cv2

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while True:
    ret, frame = video.read()

    if not ret:
        print("Camera error ❌")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    print("Faces detected:", len(faces))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("OpenCV Face Detection", frame)

    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()