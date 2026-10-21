# Arabic Sign Language Recognition Backend

A RESTful API for Arabic sign language recognition, built with **FastAPI** and structured according to **Clean Architecture** principles.

The system accepts a sequence of sign-language images, extracts and preprocesses visual landmarks, and passes the resulting feature sequence to a trained deep learning model to generate a sign prediction with a confidence score.

## Overview

The backend separates API handling, application logic, domain models, preprocessing, and model inference into distinct components. This structure promotes separation of concerns and makes the system easier to maintain, test, and extend.

## Features

- **Sign Recognition API:** Exposes an HTTP endpoint for sign prediction using FastAPI.
- **Sequence-Based Input:** Accepts multiple images representing a sign sequence.
- **Landmark Extraction:** Uses MediaPipe Holistic to extract pose and hand landmarks from input images.
- **Sequence Preprocessing:** Normalizes landmark coordinates and resamples sequences to a fixed length of 40 frames.
- **Deep Learning Inference:** Uses a trained TensorFlow/Keras model to classify the processed sequence.
- **Confidence-Based Filtering:** Returns `Unknown` when the prediction confidence falls below the configured threshold.
- **Input Validation:** Validates the input sequence and prediction response data.

## Project Architecture

The backend follows Clean Architecture principles, separating the system into distinct layers to improve maintainability and separation of concerns.

- **Presentation Layer:** Handles HTTP requests, API routing, dependency configuration, request validation, and response schemas.
- **Application Layer:** Contains the `PredictSignUseCase`, which coordinates the sign prediction workflow through an abstract predictor interface.
- **Domain Layer:** Defines the core entities used to represent sign-image sequences and prediction results.
- **Infrastructure Layer:** Implements model inference using TensorFlow and MediaPipe.
- **Utilities:** Provides landmark normalization and sequence resampling functions used during preprocessing.

## Project Structure

```text
SLT-BETWEEN-DEAF-MUTE-BACKEND/
├── presentation/
│   ├── dependencies.py
│   ├── routers.py
│   └── schemas.py
├── src/
│   ├── application/
│   │   ├── interfaces/
│   │   │   └── predictor_interface.py
│   │   └── usecase/
│   │       └── predict_sign_usecase.py
│   ├── Domain/
│   │   └── entities.py
│   ├── infrastructure/
│   │   └── predictor/
│   │       └── tensorflowpredictor.py
│   ├── utils/
│   │   └── preprocessing.py
│   └── assets/
│       └── model.h5
├── config.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```
## Installation & Setup

### Prerequisites

- Python version compatible with the pinned dependencies.
- pip package manager.
- Git.

### 1. Clone the repository

```bash
git clone https://github.com/maryhussain/SLT-between-deaf-mute-Backend.git
cd SLT-BETWEEN-DEAF-MUTE-BACKEND
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the application

```bash
uvicorn main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```
## API Reference

### Predict Sign

Predicts an Arabic sign from a sequence of uploaded images.

- **Endpoint:** `/predict`
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`

#### Request Parameters

| Parameter | Type | Description |
|---|---|---|
| `files` | List of files | Image files representing a sign sequence. |

#### Example Request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -F "files=@frame1.jpg" \
  -F "files=@frame2.jpg" \
  -F "files=@frame3.jpg"
```

Replace the example filenames with paths to your own image files.

#### Example Response

```json
{
  "label": "أريد فتح النافذة",
  "confidence": 0.97
}
```

The response contains:

- `label`: The predicted sign label. The application may return `Unknown` when the prediction confidence is below the configured threshold.
- `confidence`: A numeric confidence score constrained to the range `[0.0, 1.0]`.

#### Error Responses

| Status Code | Description |
|---|---|
| `400 Bad Request` | No files were uploaded or the input caused a `ValueError`. |
| `500 Internal Server Error` | An unexpected error occurred during prediction. |


## Configuration

The application uses a centralized configuration module to define model settings and inference parameters.

Key settings include:

- `NUM_FRAMES`: Number of frames used for each input sequence (40).
- `NUM_FEATURES`: Number of extracted features per frame (225).
- `NUM_CLASSES`: Number of supported sign classes (12).
- `CONFIDENCE_THRESHOLD`: Minimum confidence required to return a predicted sign (0.8).
- `MODEL_PATH`: Path to the trained model file.

## Limitations

- The model supports a fixed set of 12 sign classes.
- Predictions depend on the quality and content of the input images.
- The input sequence is resampled to a fixed length before inference.
- The system returns `Unknown` when the prediction confidence is below the configured threshold.


## Future Improvements

- **Bidirectional Sign Language Translation:** Extend the system to support translation in both directions between Arabic sign language and written or spoken Arabic.
- **Expanded Sign Vocabulary:** Increase the number of supported signs to cover a wider range of Arabic sign language expressions and vocabulary.

## Tech Stack

- **Python:** Core programming language.
- **FastAPI:** Building the REST API and handling HTTP requests.
- **TensorFlow / Keras:** Loading and running the trained deep learning model.
- **MediaPipe Holistic:** Extracting hand and pose landmarks from input images.
- **Pydantic:** Validating API response data.
- **Uvicorn:** Running the FastAPI application.
- **NumPy:** Numerical operations and landmark preprocessing.
- **Git & GitHub:** Version control and source code hosting.

## Author
**Eng.Mary Hussain**