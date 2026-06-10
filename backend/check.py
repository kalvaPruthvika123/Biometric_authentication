from PIL import Image

img = Image.open(r"dataset\1\face\IMG_20260226_150614726.jpg")

print("Mode:", img.mode)
print("Size:", img.size)