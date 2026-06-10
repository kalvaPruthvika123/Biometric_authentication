from fingerprint_module import match_fingerprint

user_id = "1"
test_img_path = "dataset/1/finger/1__M_Right_thumb_finger.BMP"

result = match_fingerprint(user_id, test_img_path)

print("Result:", result)
