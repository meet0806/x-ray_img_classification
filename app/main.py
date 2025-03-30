from fastapi import FastAPI, File, UploadFile, Request
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
from utils.image_processing import load_image, preprocess_image, predict
import torch
from torchvision import models
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Set up templates for rendering HTML
# templates = Jinja2Templates(directory="templates")

# Load the pre-trained model
model_path = "utils/models/resnet18_chestxrays.pth"
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 1)  # Adjust the final layer for binary classification
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))


# app.include_router(predict_router, prefix="/api")
# @app.get("/", response_class=HTMLResponse)
# async def home(request: Request):
#     """
#     Render the home page with a file upload form.
#     """
#     return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/predict/")
async def classify_xray(file: UploadFile = File(...)):
    """
    Endpoint to classify an uploaded X-ray image.
    """
    # Load and preprocess the image
    image = load_image(file.file)
    image_tensor = preprocess_image(image)

    # Make a prediction
    prediction = predict(model, image_tensor)

    # Return the result
    result = "Pneumonia" if prediction == 1 else "Normal"
    print(f"Prediction: {result}")
    return {"prediction": result}