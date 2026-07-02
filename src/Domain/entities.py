from config import Config


class SignImageSequence:
    def __init__(self, images_bytes: list[bytes]):
        if not images_bytes: 
            raise ValueError("Image list cannot be empty.")
        
        if len(images_bytes) != Config.NUM_FRAMES:
            raise ValueError(f"Image list cannot exceed {Config.NUM_FRAMES} images.")
        
        self._sequence = images_bytes

    @property  
    def images_bytes(self) -> list[bytes]:
        return self._sequence   


class PredictionResult:
    def __init__(self, label: str, confidence: float): 
        if not label:
            raise ValueError("Label cannot be empty.")
        
        if not (0.0 <= confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0.")
        
        self._label = label
        self._confidence = confidence

    @property
    def label(self):
        return self._label

    @property
    def confidence(self):
        return self._confidence