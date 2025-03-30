import os
from PIL import Image
import torch
from torchvision import transforms

# Define the transformations to apply to the images
transform_mean = [0.485, 0.456, 0.406]
transform_std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=transform_mean, std=transform_std)
])

def load_image(image_path):
    """Load an image from the specified path."""
    image = Image.open(image_path).convert("RGB")
    return image

def preprocess_image(image):
    """Preprocess the image for the model."""
    print("its here")
    return transform(image).unsqueeze(0)  # Add batch dimension

def predict(model, image_tensor):
    """Make a prediction using the model."""
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        outputs = model(image_tensor)
        preds = torch.sigmoid(outputs).round()  # Round to get binary predictions
    return preds.item()  # Return the prediction as a scalar value