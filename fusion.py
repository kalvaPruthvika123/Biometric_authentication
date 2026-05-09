from face_module import recognize_face
from fingerprint_module import match_fingerprint
from tkinter import filedialog

def authenticate():
    print("Starting Face Recognition...")
    user = recognize_face()

    if not user:
        print("Face not recognized ❌")
        return

    print(f"Face matched: User {user}")

    file_path = filedialog.askopenfilename(title="Select fingerprint image")

    if not file_path:
        print("No fingerprint selected ❌")
        return

    print("Checking fingerprint...")

    if match_fingerprint(user, file_path):
        print("Authentication Successful ✅")
    else:
        print("Authentication Failed ❌")