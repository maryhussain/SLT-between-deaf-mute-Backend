from pathlib import Path


class Config:

    #Root directory path
    BASE_DIR = Path(__file__).resolve().parent

    #Static variables
    ACTIONS = ACTIONS = [
    'bathroom','close','door','eat',
    'enter','fan','heater','open',
    'sleep','television','weakup','window'
    ]
    NUM_CLASSES = len(ACTIONS)
    NUM_FRAMES = 40
    NUM_FEATURES = 1530
    CONFIDIENCE_THRESHOLD = 0.8

    #Directory paths
    ASSET_PATH = BASE_DIR / 'assets'
    MODEL_PATH = ASSET_PATH / 'model.tflite' 
    SCALER_PATH = ASSET_PATH / 'scaler.pkl'








