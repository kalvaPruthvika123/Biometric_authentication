import os

import cv2
import face_recognition
import numpy as np
import time


DATASET_PATH = "dataset"

# face_recognition's common tolerance is 0.60. Authentication should be stricter.
FACE_DISTANCE_THRESHOLD = 0.45
FACE_MATCH_MARGIN = 0.06
MAX_FACE_IMAGE_DIM = 1000
CAMERA_TIMEOUT_SECONDS = 12
_FACE_ENCODING_CACHE = None


def _load_bgr_image(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def _resize_for_encoding(img, max_dim=MAX_FACE_IMAGE_DIM):
    height, width = img.shape[:2]
    scale = min(1.0, max_dim / max(height, width))
    if scale < 1.0:
        img = cv2.resize(
            img,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    return img


def _to_rgb_uint8(img):
    img = _resize_for_encoding(img)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def _largest_face(locations):
    return max(
        locations,
        key=lambda loc: max(0, loc[2] - loc[0]) * max(0, loc[1] - loc[3]),
    )


def _encode_face(img):
    rgb = _to_rgb_uint8(img)
    locations = face_recognition.face_locations(
        rgb,
        number_of_times_to_upsample=1,
        model="hog",
    )

    if not locations:
        return None

    location = _largest_face(locations)
    encodings = face_recognition.face_encodings(
        rgb,
        known_face_locations=[location],
        num_jitters=1,
    )
    if not encodings:
        return None
    return encodings[0]


def _load_dataset_encodings():
    global _FACE_ENCODING_CACHE
    if _FACE_ENCODING_CACHE is not None:
        return _FACE_ENCODING_CACHE

    known = []

    for user_id in sorted(os.listdir(DATASET_PATH)):
        face_folder = os.path.join(DATASET_PATH, user_id, "face")
        if not os.path.isdir(face_folder):
            continue

        for img_name in sorted(os.listdir(face_folder)):
            img_path = os.path.join(face_folder, img_name)
            try:
                encoding = _encode_face(_load_bgr_image(img_path))
            except Exception as exc:
                print(f"  Error loading face {img_path}: {exc}")
                continue

            if encoding is None:
                print(f"  No face found in dataset image: {img_path}")
                continue

            known.append(
                {
                    "user_id": user_id,
                    "image": img_name,
                    "encoding": encoding,
                }
            )

    _FACE_ENCODING_CACHE = known
    return known


def _capture_camera_face():
    print("Opening camera...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not opening")
        return None

    print("Look at the camera. Press ESC to cancel.")
    print(f"Camera will timeout after {CAMERA_TIMEOUT_SECONDS} seconds if no face is found.")
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    captured = None
    stable_face_frames = 0
    start_time = time.monotonic()
    cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)

    while True:
        if time.monotonic() - start_time > CAMERA_TIMEOUT_SECONDS:
            print("Face scan timed out")
            break

        ret, frame = cap.read()
        if not ret:
            print("Camera error")
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            stable_face_frames += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if stable_face_frames >= 15:
                captured = frame.copy()
                cv2.imshow("Face Recognition", frame)
                cv2.waitKey(300)
                break
        else:
            stable_face_frames = 0

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured


def recognize_face():
    captured_face = _capture_camera_face()
    if captured_face is None:
        print("No face captured")
        return None

    print("Face captured. Loading dataset encodings...")
    known = _load_dataset_encodings()
    if not known:
        print("No usable face encodings found in dataset")
        return None

    probe_encoding = _encode_face(captured_face)
    if probe_encoding is None:
        print("Captured image did not contain an encodable face")
        return None

    encodings = [item["encoding"] for item in known]
    distances = face_recognition.face_distance(encodings, probe_encoding)

    per_user = {}
    for item, distance in zip(known, distances):
        per_user.setdefault(item["user_id"], []).append((float(distance), item["image"]))

    ranked = []
    print("\n" + "=" * 60)
    print("FACE MATCHING ANALYSIS")
    print("=" * 60)

    for user_id, user_scores in per_user.items():
        user_scores.sort(key=lambda row: row[0])
        best_distance = user_scores[0][0]
        median_distance = float(np.median([row[0] for row in user_scores]))
        ranked.append((best_distance, median_distance, user_id, user_scores[0][1]))

    ranked.sort(key=lambda row: (row[0], row[1]))

    for best_distance, median_distance, user_id, best_image in ranked:
        print(
            f"User {user_id}: best={best_distance:.4f}, "
            f"median={median_distance:.4f}, best image={best_image}"
        )

    best_distance, median_distance, best_user, best_image = ranked[0]
    runner_up_distance = ranked[1][0] if len(ranked) > 1 else 1.0
    margin = runner_up_distance - best_distance

    print("\n--- FACE DECISION ---")
    print(f"Best user: {best_user}")
    print(f"Best distance: {best_distance:.4f}")
    print(f"Runner-up distance: {runner_up_distance:.4f}")
    print(f"Margin: {margin:.4f}")
    print(f"Threshold: {FACE_DISTANCE_THRESHOLD}")

    if best_distance <= FACE_DISTANCE_THRESHOLD and margin >= FACE_MATCH_MARGIN:
        print(f"FACE RECOGNIZED -> User {best_user}")
        return best_user

    if best_distance > FACE_DISTANCE_THRESHOLD:
        print("FACE NOT RECOGNIZED: best distance is above threshold")
    else:
        print("FACE NOT RECOGNIZED: best match is not clearly separated")
    return None
