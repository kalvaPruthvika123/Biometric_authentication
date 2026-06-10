import { useState } from "react";

function App() {
  const [faceFile, setFaceFile] = useState(null);
  const [fingerprintFile, setFingerprintFile] = useState(null);
  const [facePreview, setFacePreview] = useState(null);
  const [fingerprintPreview, setFingerprintPreview] = useState(null);

  const [status, setStatus] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const [evaluation, setEvaluation] = useState(null);

  const handleFileChange = (event, setFile, setPreview) => {
    const file = event.target.files?.[0] ?? null;

    setError("");
    setResult(null);

    if (!file) {
      setFile(null);
      setPreview(null);
      return;
    }

    const extension = file.name.split(".").pop().toLowerCase();

    if (!["jpg", "jpeg", "png", "bmp"].includes(extension)) {
      setError("Please upload JPG, JPEG, PNG, or BMP images.");
      return;
    }

    setFile(file);
    setPreview(URL.createObjectURL(file));
  };

  const verifyIdentity = async () => {
    if (!faceFile || !fingerprintFile) {
      setError("Please upload both a face image and fingerprint image.");
      return;
    }

    setStatus("Verifying identity...");
    setError("");
    setResult(null);

    const formData = new FormData();

    formData.append("face_image", faceFile);
    formData.append("fingerprint_image", fingerprintFile);

    try {
      const response = await fetch("/api/authenticate", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        setError(data.message || "Request failed.");
        setStatus("");
        return;
      }

      setResult(data);

      setStatus(
        data.authenticated
          ? "Authentication successful."
          : "Authentication failed."
      );
    } catch (err) {
      setError("Unable to connect to backend.");
      setStatus("");
    }
  };

  const loadEvaluation = async () => {
    try {
      const response = await fetch("/api/evaluation");
      const data = await response.json();
      setEvaluation(data);
    } catch (err) {
      console.error(err);
      setError("Unable to load evaluation results.");
    }
  };

  return (
    <div className="app-shell">
      <header>
        <h1>Biometric Authentication</h1>
        <p>Upload face and fingerprint images to verify identity.</p>
      </header>

      <main>
        <section className="upload-section">
          <div className="upload-card">
            <label>Upload Face Image</label>

            <input
              type="file"
              accept="image/*,.bmp,.BMP"
              onChange={(e) =>
                handleFileChange(
                  e,
                  setFaceFile,
                  setFacePreview
                )
              }
            />

            {facePreview ? (
              <img
                className="preview-image"
                src={facePreview}
                alt="Face"
              />
            ) : (
              <div className="preview-placeholder">
                Face Preview
              </div>
            )}
          </div>

          <div className="upload-card">
            <label>Upload Fingerprint Image</label>

            <input
              type="file"
              accept="image/*,.bmp,.BMP"
              onChange={(e) =>
                handleFileChange(
                  e,
                  setFingerprintFile,
                  setFingerprintPreview
                )
              }
            />

            {fingerprintPreview ? (
              <img
                className="preview-image"
                src={fingerprintPreview}
                alt="Fingerprint"
              />
            ) : (
              <div className="preview-placeholder">
                Fingerprint Preview
              </div>
            )}
          </div>
        </section>

        <button
          className="verify-button"
          onClick={verifyIdentity}
        >
          Verify Identity
        </button>

        <section className="result-section">
          <h2>Authentication Result</h2>

          {status && <p>{status}</p>}
          {error && <p>{error}</p>}

          {result && (
            <div className="result-card">
              <p>
                <strong>User ID:</strong> {result.user_id}
              </p>

              <p>
                <strong>Face Confidence:</strong>{" "}
                {Math.round(
                  (result.face_result?.confidence || 0) * 100
                )}
                %
              </p>

              <p>
                <strong>Face Recognized:</strong>{" "}
                {result.face_result?.recognized
                  ? "Yes"
                  : "No"}
              </p>

              <p>
                <strong>Fingerprint Match:</strong>{" "}
                {result.fingerprint_match
                  ? "Yes"
                  : "No"}
              </p>

              {result.message && (
                <p>{result.message}</p>
              )}
            </div>
          )}
        </section>

        <hr />

        <section>
          <h2>Model Evaluation Dashboard</h2>

          <button onClick={loadEvaluation}>
            Load Evaluation Results
          </button>

          {evaluation && (
            <>
              <h3>Evaluation Report</h3>

              <pre
                style={{
                  whiteSpace: "pre-wrap",
                  textAlign: "left",
                }}
              >
                {evaluation.report}
              </pre>

              <h3>Confusion Matrix</h3>
              <img
                src={evaluation.confusion_matrix}
                alt="Confusion Matrix"
                style={{ maxWidth: "100%" }}
              />

              <h3>Accuracy Graph</h3>
              <img
                src={evaluation.accuracy_graph}
                alt="Accuracy"
                style={{ maxWidth: "100%" }}
              />

              <h3>Precision Graph</h3>
              <img
                src={evaluation.precision_graph}
                alt="Precision"
                style={{ maxWidth: "100%" }}
              />

              <h3>Recall Graph</h3>
              <img
                src={evaluation.recall_graph}
                alt="Recall"
                style={{ maxWidth: "100%" }}
              />

              <h3>F1 Score Graph</h3>
              <img
                src={evaluation.f1_graph}
                alt="F1"
                style={{ maxWidth: "100%" }}
              />
            </>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;