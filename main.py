import os
os.environ["KERAS_BACKEND"] = "numpy"
from fastapi import FastAPI
from presentation.routers import router as prediction_router

app = FastAPI()
app.include_router(prediction_router)


