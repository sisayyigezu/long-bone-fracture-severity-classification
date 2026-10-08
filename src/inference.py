from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

from src.model import CustomCNN


# Class order must match ImageFolder training
CLASS_NAMES = [
    "Healthy",
    "Simple",
    "Wedge_Complex",
]

IMAGE_SIZE = 224

# Same evaluation preprocessing used during your trained model's testing
TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model(model_path: str):
    """Load the trained Custom CNN weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CustomCNN(num_classes=len(CLASS_NAMES))
    model.load_state_dict(
        torch.load(model_path, map_location=device)
    )

    model.to(device)
    model.eval()

    return model, device


def predict(image_path: str, model_path: str):
    """Predict the fracture severity class for one X-ray image."""

    model, device = load_model(model_path)

    image = Image.open(image_path).convert("RGB")
    image_tensor = TRANSFORM(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

    predicted_index = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0, predicted_index].item()

    probabilities_dict = {
        CLASS_NAMES[i]: float(probabilities[0, i])
        for i in range(len(CLASS_NAMES))
    }

    return {
        "prediction": CLASS_NAMES[predicted_index],
        "confidence": confidence,
        "probabilities": probabilities_dict,
    }