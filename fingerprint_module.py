import os

import cv2
import numpy as np


DATASET_PATH = "dataset"

# SIFT ratio-match counts from this dataset:
# exact enrolled image ~= 800, strongest wrong-person image observed ~= 20.
FINGERPRINT_MIN_GOOD_MATCHES = 25
FINGERPRINT_MATCH_MARGIN = 8
FINGERPRINT_RATIO_TEST = 0.72


def preprocess(img):
    img = cv2.resize(img, (300, 300))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def extract_features(img):
    sift = cv2.SIFT_create(
        nfeatures=800,
        contrastThreshold=0.02,
        edgeThreshold=10,
    )
    return sift.detectAndCompute(img, None)


def compare_fingerprints(img1, img2):
    _, desc1 = extract_features(img1)
    _, desc2 = extract_features(img2)

    if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
        return 0

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(desc1, desc2, k=2)
    good_matches = [
        match
        for match, second_best in matches
        if match.distance < FINGERPRINT_RATIO_TEST * second_best.distance
    ]
    return len(good_matches)


def _iter_dataset_fingerprints():
    for user_id in sorted(os.listdir(DATASET_PATH)):
        finger_folder = os.path.join(DATASET_PATH, user_id, "finger")
        if not os.path.isdir(finger_folder):
            continue

        for img_name in sorted(os.listdir(finger_folder)):
            img_path = os.path.join(finger_folder, img_name)
            db_img_raw = cv2.imread(img_path)
            if db_img_raw is None:
                print(f"  Could not read fingerprint image: {img_path}")
                continue

            yield user_id, img_name, preprocess(db_img_raw)


def identify_fingerprint(test_img_path):
    test_img_raw = cv2.imread(test_img_path)
    if test_img_raw is None:
        print("Could not read fingerprint image")
        return None

    test_img = preprocess(test_img_raw)
    scores = []

    for user_id, img_name, db_img in _iter_dataset_fingerprints():
        score = compare_fingerprints(test_img, db_img)
        scores.append((score, user_id, img_name))

    if not scores:
        print("No fingerprint images available in dataset")
        return None

    scores.sort(key=lambda row: row[0], reverse=True)
    best_score, best_user, best_image = scores[0]
    runner_up_score = scores[1][0] if len(scores) > 1 else 0
    margin = best_score - runner_up_score

    print("\n" + "=" * 60)
    print("FINGERPRINT MATCHING ANALYSIS")
    print("=" * 60)
    for score, user_id, img_name in scores[:8]:
        print(f"User {user_id}: matches={score}, image={img_name}")

    print("\n--- FINGERPRINT DECISION ---")
    print(f"Best user: {best_user}")
    print(f"Best score: {best_score}")
    print(f"Runner-up score: {runner_up_score}")
    print(f"Margin: {margin}")
    print(f"Minimum good matches: {FINGERPRINT_MIN_GOOD_MATCHES}")

    if best_score < FINGERPRINT_MIN_GOOD_MATCHES:
        print("FINGERPRINT NOT RECOGNIZED: match count is below threshold")
        return None

    if margin < FINGERPRINT_MATCH_MARGIN:
        print("FINGERPRINT NOT RECOGNIZED: best match is not clearly separated")
        return None

    print(f"FINGERPRINT RECOGNIZED -> User {best_user}")
    return best_user


def match_fingerprint(user_id, test_img_path):
    matched_user = identify_fingerprint(test_img_path)
    if matched_user is None:
        return False

    if str(matched_user) != str(user_id):
        print(
            "FINGERPRINT USER MISMATCH: "
            f"face user={user_id}, fingerprint user={matched_user}"
        )
        return False

    print(f"FINGERPRINT MATCHED with face user {user_id}")
    return True
