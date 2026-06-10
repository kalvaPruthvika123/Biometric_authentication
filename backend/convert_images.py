from PIL import Image
import os

input_folder = "dataset/1/face"
output_folder = "dataset/1/face_fixed"

os.makedirs(output_folder, exist_ok=True)

for img_name in os.listdir(input_folder):
    input_path = os.path.join(input_folder, img_name)

    try:
        img = Image.open(input_path)

        # Force clean RGB 8-bit format
        img = img.convert("RGB")

        output_path = os.path.join(output_folder, img_name)
        img.save(output_path, "JPEG")

        print("Converted:", img_name)

    except Exception as e:
        print("Failed:", img_name, e)