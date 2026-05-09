import cv2

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Check if cascade loaded properly
if face_cascade.empty():
    print("Error loading Haar cascade ❌")
    exit()

# Start camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera not working ❌")
        break

    frame_count += 1

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 🔥 Improved face detection (more sensitive)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(30, 30)
    )

    print("Faces detected:", len(faces))

    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show camera
    cv2.imshow("Camera Test", frame)

    # ✅ Press ESC to stop
    if cv2.waitKey(1) == 27:
        break

    # ✅ Auto stop after ~3 seconds (safety)
    if frame_count > 100:
        break

# Release camera
cap.release()
cv2.destroyAllWindows()