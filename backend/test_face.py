from PIL import Image
import numpy as np
import face_recognition

# Load and convert
img = Image.open("dataset/1/face/IMG_20260226_150614726.jpg").convert("RGB")

# Convert to numpy
img = np.array(img)

# 🔥 FORCE correct dtype
img = img.astype('uint8')

print("Shape:", img.shape)
print("Dtype:", img.dtype)

# Face encoding
encodings = face_recognition.face_encodings(img)

print("Encodings found:", len(encodings))