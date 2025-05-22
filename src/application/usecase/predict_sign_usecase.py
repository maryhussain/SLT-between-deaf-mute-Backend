from src.Domain.entities import  PredictionResult, SignSequence
from src.application.interfaces.predictor_interface import PredictorInterface
from config import Config

class PredictSignUseCase:

    def __init__(self, predictor: PredictorInterface):
        self.predictor = predictor



    def execute(self, sign_sequence: SignSequence) -> PredictionResult:

        try:

            preprocessed_sequence = self.predictor.preprocess(sign_sequence.sequence)

            label, confidence = self.predictor.predict(preprocessed_sequence)

            if confidence < Config.CONFIDENCE_THRESHOLD :
                return PredictionResult("Unknown", confidence)
            
            return PredictionResult(label, confidence)
        
        except ValueError  :
            
            return PredictionResult("Unknown", 0.0)
        
        


        