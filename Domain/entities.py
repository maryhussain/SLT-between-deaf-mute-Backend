class SignSequence:
    def __init__(self, sequence: list):
        if not sequence: 
            raise ValueError("Sequence cannot be empty.")
        self._sequence = sequence  

    @property  
    def sequence(self):
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