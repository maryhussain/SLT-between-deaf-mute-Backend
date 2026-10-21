from pathlib import Path


class Config:

    #Root directory path
    BASE_DIR = Path(__file__).resolve().parent

    #Static variables
    ACTIONS = [ 
        "أريد الذهاب إلى الحمام" , "أريد شحن الهاتف", "أريد إغلاق الخزانة" , "شكراً لحسن استماعكم",
        "أريد شرب الماء","أريد غسل الملابس", "أريد تشغيل المروحة", "أريد تشغيل المدفأة",
        "أريد فتح الخزانة", "أريد الذهاب إلى النوم", "أريد تشغيل تلفاز","أريد فتح النافذة"
        ]

    NUM_CLASSES = len(ACTIONS)
    NUM_FRAMES = 40
    NUM_FEATURES = 225
    CONFIDENCE_THRESHOLD = 0.8

    #Directory paths
    ASSET_PATH = BASE_DIR / "src" / "assets"
    MODEL_PATH = str(ASSET_PATH  / "model.h5")
    
