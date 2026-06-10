import os
import tempfile
from flask import Flask, jsonify, request
from fusion import authenticate_face_and_fingerprint
from face_module import recognize_face
from fingerprint_module import match_fingerprint
from flask import send_from_directory

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
EVALUATION_DIR = os.path.join(PROJECT_ROOT, "evaluation_results")
# Allow JPG, JPEG, PNG, and BMP files
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _is_allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


def _save_uploaded_file(file_key):
    if file_key not in request.files:
        raise ValueError(f"Missing required file field: {file_key}")

    uploaded_file = request.files[file_key]

    if uploaded_file.filename == "":
        raise ValueError("No file was selected.")

    if not _is_allowed_file(uploaded_file.filename):
        raise ValueError(
            "Only JPG, JPEG, PNG, and BMP image files are allowed."
        )

    suffix = os.path.splitext(uploaded_file.filename)[1].lower()

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )

    uploaded_file.save(temp_file.name)
    temp_file.close()

    return temp_file.name


def _cleanup_temp_files(file_paths):
    for path in file_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


@app.route("/")
def home():
    return jsonify({
        "success": True,
        "message": "Biometric Authentication API Running"
    })


@app.route("/api/face-recognition", methods=["POST"])
def face_recognition_route():
    temp_files = []

    try:
        face_path = _save_uploaded_file("face_image")
        temp_files.append(face_path)

        recognition = recognize_face(face_path)

        return jsonify({
            "success": True,
            "face_result": recognition
        })

    except Exception as exc:
        return jsonify({
            "success": False,
            "message": str(exc)
        }), 400

    finally:
        _cleanup_temp_files(temp_files)


@app.route("/api/fingerprint-recognition", methods=["POST"])
def fingerprint_recognition_route():
    temp_files = []

    try:
        user_id = request.form.get("user_id")

        if not user_id:
            raise ValueError("Missing user_id field.")

        fingerprint_path = _save_uploaded_file("fingerprint_image")
        temp_files.append(fingerprint_path)

        fingerprint_ok = match_fingerprint(
            user_id,
            fingerprint_path
        )

        return jsonify({
            "success": True,
            "user_id": user_id,
            "fingerprint_match": fingerprint_ok
        })

    except Exception as exc:
        return jsonify({
            "success": False,
            "message": str(exc)
        }), 400

    finally:
        _cleanup_temp_files(temp_files)


@app.route("/api/authenticate", methods=["POST"])
def authenticate_route():
    temp_files = []

    try:
        face_path = _save_uploaded_file("face_image")
        fingerprint_path = _save_uploaded_file("fingerprint_image")

        temp_files.extend([
            face_path,
            fingerprint_path
        ])

        result = authenticate_face_and_fingerprint(
            face_path,
            fingerprint_path
        )

        response = {
            "success": True,
            "authenticated": result["authenticated"],
            "user_id": result["user_id"],
            "face_result": result["face_result"],
            "fingerprint_match": result["fingerprint_match"]
        }

        if not result["authenticated"]:
            if not result["face_result"]["recognized"]:
                response["message"] = "Unauthorized User"
            else:
                response["message"] = (
                    "Fingerprint did not match the predicted face user."
                )

        return jsonify(response)

    except Exception as exc:
        return jsonify({
            "success": False,
            "message": str(exc)
        }), 400

    finally:
        _cleanup_temp_files(temp_files)
@app.route("/api/evaluation")
def evaluation_data():
    report_path = os.path.join(EVALUATION_DIR, "evaluation_report.txt")

    report_text = ""

    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report_text = f.read()
        return jsonify({
    "report": report_text,
    "confusion_matrix": "http://127.0.0.1:5000/evaluation/confusion_matrix.png",
    "accuracy_graph": "http://127.0.0.1:5000/evaluation/accuracy_graph.png",
    "precision_graph": "http://127.0.0.1:5000/evaluation/precision_graph.png",
    "recall_graph": "http://127.0.0.1:5000/evaluation/recall_graph.png",
    "f1_graph": "http://127.0.0.1:5000/evaluation/f1_graph.png"
})
@app.route("/api/eval-debug")
def eval_debug():
    return jsonify({
        "evaluation_dir": EVALUATION_DIR,
        "exists": os.path.exists(EVALUATION_DIR),
        "files": os.listdir(EVALUATION_DIR) if os.path.exists(EVALUATION_DIR) else []
    })

@app.route("/evaluation/<path:filename>")
def serve_evaluation_file(filename):
    return send_from_directory(EVALUATION_DIR, filename)


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )