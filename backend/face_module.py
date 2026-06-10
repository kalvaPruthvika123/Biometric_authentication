import os

from PIL import Image
import cv2
import face_recognition
import numpy as np


DATASET_PATH = "dataset"

# face_recognition's common tolerance is 0.60. Authentication should be stricter.
FACE_DISTANCE_THRESHOLD = 0.45
FACE_MATCH_MARGIN = 0.06
MAX_FACE_IMAGE_DIM = 1000
SUPPORTED_FACE_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
_FACE_ENCODING_CACHE = None



def _load_bgr_image(path):
    try:
        pil_img = Image.open(path).convert("RGB")
        rgb = np.array(pil_img, dtype=np.uint8)

        print(
            f"Loaded {path} | shape={rgb.shape} | dtype={rgb.dtype}"
        )

        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    except Exception as exc:
        raise ValueError(f"Could not read image: {path}. {exc}")

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


def _validate_face_image_path(path):
    if not path or not os.path.exists(path):
        raise ValueError("No face image selected.")

    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_FACE_IMAGE_EXTENSIONS:
        raise ValueError("Unsupported face image file type. Please upload a JPG or PNG image.")


def recognize_face(face_image_path):
    _validate_face_image_path(face_image_path)
    probe_image = _load_bgr_image(face_image_path)

    print("Loading dataset encodings...")
    known = _load_dataset_encodings()
    if not known:
        raise RuntimeError("No usable face encodings found in dataset.")

    probe_encoding = _encode_face(probe_image)
    if probe_encoding is None:
        raise ValueError("Uploaded image did not contain an encodable face.")

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
    confidence = max(0.0, 1.0 - best_distance)
    recognized = best_distance <= FACE_DISTANCE_THRESHOLD and margin >= FACE_MATCH_MARGIN

    print("\n--- FACE DECISION ---")
    print(f"Best user: {best_user}")
    print(f"Best distance: {best_distance:.4f}")
    print(f"Runner-up distance: {runner_up_distance:.4f}")
    print(f"Margin: {margin:.4f}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Threshold: {FACE_DISTANCE_THRESHOLD}")

    return {
        "user_id": best_user,
        "distance": best_distance,
        "confidence": confidence,
        "recognized": recognized,
        "best_image": best_image,
        "margin": margin,
    }
