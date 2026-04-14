# Sign Language Recognition Backend

A backend system for translating sign language into text using a trained machine learning model.
Built with a clean and scalable architecture using FastAPI and TFLite.

## Overview

This project aims to facilitate communication between deaf and mute individuals and others by recognizing hand gestures and converting them into meaningful text.

The system receives a sequence of keypoints extracted from video frames and predicts the corresponding sign.

## Features

- Real-time sign prediction
- Lightweight model inference using TFLite
- Clean Architecture (scalable & maintainable)
- Separation of concerns (Domain, Application, Infrastructure, API)
- Easy integration with frontend applications

## Tech Stack
- **FastAPI**
- **TensorFlow Lite (TFLite)**
- **NumPy**
- **Scikit-learn**
- **Python**

## Workflow

- The frontend captures video
- Video is converted into 40 frames
- Keypoints are extracted and sent as a sequence
- Backend receives the sequence via API
- Data is preprocessed and scaled
- The model predicts the sign
- The predicted label is returned to the frontend

## API Endpoint
**POST /predict**
Request:
{
  "sequence": [...]
}
Response:
{
  "prediction": "hello"
}

## How to Run
1. `pip install -r requirements.txt`
2. `uvicorn main:app --reload`

## Key Design Decisions
- FastAPI for high-performance APIs
- TFLite for fast and lightweight inference
- Clean Architecture for scalability and maintainability
- Decoupling model logic from business logic

## Future Improvements

- Deploy the model on edge devices
- Improve prediction accuracy
- Add authentication system

## Author

**Eng.Mary Hussain**

## Notes

**This project demonstrates practical implementation of:**

- Machine Learning deployment
- Backend architecture design
- Clean coding practices