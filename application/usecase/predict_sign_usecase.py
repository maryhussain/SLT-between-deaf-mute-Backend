from Domain.entities import  PredictionResult, SignSequence
from application.interfaces.predictor_interface import PredictorInterface
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
        
        except ValueError as e :
            
            return PredictionResult("Unknown", 0.0)
        
        


        