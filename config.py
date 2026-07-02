from pathlib import Path


class Config:

    #Root directory path
    BASE_DIR = Path(__file__).resolve().parent

    #Static variables
    ACTIONS = ACTIONS = [ 
        "الحمام" , "اشحن الهاتف", "أغلق" , "باب",
        "يشرب متة","أدخل", "مروحة", "مدفأة",
        "يفتح", "ينام", "تلفاز","نافذة"
        ]

    NUM_CLASSES = len(ACTIONS)
    NUM_FRAMES = 40
    NUM_FEATURES = 225
    CONFIDENCE_THRESHOLD = 0.8

    #Directory paths
    ASSET_PATH = BASE_DIR / 'assets'
    MODEL_PATH = str(ASSET_PATH / 'model.h5')
    SCALER_PATH = ASSET_PATH / 'scaler.pkl'
