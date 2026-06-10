from face_module import recognize_face
from fingerprint_module import match_fingerprint

def authenticate_face_and_fingerprint(face_image_path, fingerprint_image_path):
    recognition = recognize_face(face_image_path)
    fingerprint_ok = match_fingerprint(recognition["user_id"], fingerprint_image_path)

    return {
        "user_id": recognition["user_id"],
        "face_result": recognition,
        "fingerprint_match": fingerprint_ok,
        "authenticated": recognition["recognized"] and fingerprint_ok,
    }
