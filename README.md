# X-ray Image Classification App

This project is a web-based application for classifying chest X-ray images as either "Pneumonia" or "Normal". It consists of a **React frontend** for uploading images and displaying results, and a **FastAPI backend** for processing the images using a pre-trained deep learning model.

---

## Features

- Upload chest X-ray images via the web interface.
- Preview the uploaded image before submission.
- Get predictions (e.g., "Pneumonia" or "Normal") from the backend.
- Fully containerized using Docker for easy deployment.

![Demo](https://github.com/meet0806/x-ray_img_classification/blob/master/Screenshot%202025-03-30%20164752.png)

---

## Prerequisites

- **Docker**: Install [Docker](https://www.docker.com/).
- **Docker Compose**: Install [Docker Compose](https://docs.docker.com/compose/).

---

## Setup Instructions (Local)

### 1. Clone the Repository
```bash
git clone --recurse-submodules <main-repo-url>
cd x-ray_img_classification
```

### 2. Build and Run the Containers
Use Docker Compose to build and start the frontend and backend services:

```bash
docker-compose build
docker-compose up
```

### 3. Acess the Application
- Frontend: Open your browser and go to http://localhost:3000.
- Backend: The FastAPI backend is available at http://localhost:8000

---

## Usage

1. Open the frontend in your browser (http://localhost:3000).
2. Upload a chest X-ray image using the file input field.
3. Preview the uploaded image.
4. Click the "Classify" button to send the image to the backend for prediction.
5. View the prediction result (e.g., "Pneumonia" or "Normal") displayed on the page.

---

## API Endpoints 

/api/predict/ (POST)
- Description: Accepts an uploaded X-ray image and returns the classification result.
- Request:
  - file: The uploaded image file (multipart/form-data).
- Response:
```JSON
{
  "prediction": "Pneumonia"
}
```
---

## Technology Used

#### Frontend
- React: For building the UI
- Bootstrap: For Styling UI
#### Backend
- FastAPI: For building the REST API.
- PyTorch: For loading and using the pre-trained model.
- Pillow: For image processing.
#### Deployment
- Docker: For containerizing the application.
- Docker Compose: For managing multi-container applications.
