import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# --------------------------------------------------
# Add project root to Python path
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.inference import predict


# --------------------------------------------------
# Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Long Bone Fracture Severity Classification API",
    description="API for Custom CNN-based long bone fracture severity classification.",
    version="1.0.0"
)


# --------------------------------------------------
# Allow frontend to communicate with backend
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Model path
# --------------------------------------------------

MODEL_PATH = str(
    PROJECT_ROOT / "models" / "CustomCNN_model.pth"
)


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Long Bone Fracture Severity Classification API",
        "status": "running"
    }


# --------------------------------------------------
# Health check endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
async def predict_fracture(file: UploadFile = File(...)):

    # Check file type
    allowed_types = {
        "image/jpeg",
        "image/png",
        "image/bmp",
        "image/tiff"
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid X-ray image (JPG, PNG, BMP, or TIFF)."
        )

    try:
        # Read uploaded image
        file_bytes = await file.read()

        # Save temporarily because predict() currently accepts an image path
        with tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False
        ) as temp_file:

            temp_file.write(file_bytes)
            temp_path = temp_file.name

        # Run model prediction
        result = predict(
            temp_path,
            MODEL_PATH
        )

        # Delete temporary file
        Path(temp_path).unlink(missing_ok=True)

        return {
            "filename": file.filename,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "probabilities": result["probabilities"]
        }

    except Exception as e:

        # Clean up temporary file if necessary
        if "temp_path" in locals():
            Path(temp_path).unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )