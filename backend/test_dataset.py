import os

print("START")

print("Checking dataset path...")

if not os.path.exists("dataset"):
    print("❌ dataset folder not found")
else:
    users = os.listdir("dataset")
    print("Users:", users)

    for user in users:
        print("User:", user)

        face_path = f"dataset/{user}/face"
        finger_path = f"dataset/{user}/finger"

        print("Face path exists:", os.path.exists(face_path))
        print("Finger path exists:", os.path.exists(finger_path))

        if os.path.exists(face_path):
            print("Face files:", os.listdir(face_path))
        if os.path.exists(finger_path):
            print("Finger files:", os.listdir(finger_path))

        print("------")