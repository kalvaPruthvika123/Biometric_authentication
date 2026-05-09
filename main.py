import tkinter as tk
from tkinter import filedialog
from face_module import recognize_face
from fingerprint_module import match_fingerprint

def authenticate():
    result_label.config(text="Scanning face...")

    user = recognize_face()

    if not user:
        result_label.config(text="Face not recognized ❌")
        return

    result_label.config(text=f"Face matched: User {user}")

    file_path = filedialog.askopenfilename(title="Select fingerprint image")

    if not file_path:
        result_label.config(text="No fingerprint selected ❌")
        return

    if match_fingerprint(user, file_path):
        result_label.config(text=f"Authentication Successful ✅ (User {user})")
    else:
        result_label.config(text="Authentication Failed ❌")


# UI
root = tk.Tk()
root.title("Biometric Authentication System")
root.geometry("400x200")

btn = tk.Button(root, text="Authenticate", command=authenticate, height=2, width=20)
btn.pack(pady=20)

result_label = tk.Label(root, text="", font=("Arial", 12))
result_label.pack()

root.mainloop()