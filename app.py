import os
import uuid
from pathlib import Path

from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename

from model import TrafficSignClassifier


# --------------------------------------------------
# Flask Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"

UPLOAD_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB


# --------------------------------------------------
# Load Trained Traffic Sign Model
# --------------------------------------------------

MODEL_PATH = BASE_DIR / "model.joblib"

classifier = TrafficSignClassifier(model_path=str(MODEL_PATH))


# --------------------------------------------------
# Helper Functions
# --------------------------------------------------

def allowed_file(filename):
    """Check whether the uploaded file has a supported extension."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# --------------------------------------------------
# Home Page
# --------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# --------------------------------------------------
# Prediction API
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if "image_file" not in request.files:
        return jsonify({
            "success": False,
            "error": "No image file was uploaded."
        }), 400

    file = request.files["image_file"]

    if file.filename == "":
        return jsonify({
            "success": False,
            "error": "Please select an image."
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": "Unsupported image format. Please use PNG, JPG, JPEG, WEBP, or BMP."
        }), 400

    try:
        # Create a safe unique filename
        original_name = secure_filename(file.filename)
        extension = original_name.rsplit(".", 1)[1].lower()

        unique_filename = f"{uuid.uuid4().hex}.{extension}"

        file_path = UPLOAD_FOLDER / unique_filename

        # Save uploaded image
        file.save(str(file_path))

        # Run the trained classifier
        result = classifier.predict(str(file_path))

        # Add image URL for the frontend
        result["image_url"] = url_for(
            "uploaded_file",
            filename=unique_filename
        )

        result["success"] = True

        return jsonify(result)

    except Exception as e:

        print("Prediction error:", e)

        return jsonify({
            "success": False,
            "error": f"Prediction failed: {str(e)}"
        }), 500


# --------------------------------------------------
# Serve Uploaded Images
# --------------------------------------------------

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    from flask import send_from_directory

    return send_from_directory(
        app.config["UPLOAD_FOLDER"],
        filename
    )


# --------------------------------------------------
# Sample Images
# --------------------------------------------------

@app.route("/samples/<filename>")
def sample_image(filename):

    samples_folder = BASE_DIR / "samples"

    return send_from_directory(
        str(samples_folder),
        filename
    )


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.route("/health")
def health():

    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL_PATH.exists()
    })


# --------------------------------------------------
# Run Application
# --------------------------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )