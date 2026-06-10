# Biometric Authentication

This project is now split into a Python backend and a React frontend.
The existing face recognition and fingerprint recognition logic remain unchanged.

## Folder structure

```
biometric-authentication/
├── backend/
│   ├── app.py
│   ├── database.py
│   ├── face_module.py
│   ├── fingerprint_module.py
│   ├── fusion.py
│   ├── main.py
│   ├── requirements.txt
│   ├── dataset/
│   ├── convert_images.py
│   ├── test_face.py
│   ├── test_fingerprint.py
│   ├── test_dataset.py
│   ├── test_camera_face.py
│   └── test_opencv_face.py
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx
        ├── App.jsx
        └── index.css
```

## How the code is reorganized

- `backend/` contains the existing Python recognition logic plus the new Flask API layer.
- `frontend/` contains the React application that replaces the Tkinter UI.
- `backend/app.py` exposes API routes for face recognition, fingerprint recognition, and combined authentication.
- `backend/fusion.py` now exposes `authenticate_face_and_fingerprint()` to connect the two subsystems.
- `backend/main.py` is now a placeholder entrypoint and no longer contains Tkinter UI.

## Backend setup

Open PowerShell in `c:\biometric_authentication\backend` and run:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

To start the backend API:

```powershell
python app.py
```

The backend listens on `http://localhost:5000`.

## Frontend setup

Open PowerShell in `c:\biometric_authentication\frontend` and run:

```powershell
npm install
npm run dev
```

This starts the React frontend on `http://localhost:5173`.
The Vite dev server is configured to proxy `/api` requests to the backend.

## Usage

1. Open `http://localhost:5173` in your browser.
2. Upload a face image (`.jpg`, `.jpeg`, or `.png`).
3. Upload a fingerprint image (`.jpg`, `.jpeg`, or `.png`).
4. Click **Verify Identity**.
5. The frontend shows the authentication result and confidence score.

## API exposure

The backend exposes the Python recognition functions as HTTP endpoints:

- `POST /api/face-recognition`
  - Upload a `face_image` file.
  - Returns face prediction and confidence.

- `POST /api/fingerprint-recognition`
  - Upload a `fingerprint_image` file and `user_id` form field.
  - Returns whether the fingerprint matches the claimed user.

- `POST /api/authenticate`
  - Upload both `face_image` and `fingerprint_image` together.
  - The server uses `fusion.authenticate_face_and_fingerprint()`.
  - Returns the final authentication result.

## Notes

- The existing recognition logic in `backend/face_module.py`, `backend/fingerprint_module.py`, and `backend/fusion.py` is unchanged except for a small API wrapper.
- All Tkinter UI code has been removed from the project.
- The React frontend handles file upload, previews, verification, and result display.
