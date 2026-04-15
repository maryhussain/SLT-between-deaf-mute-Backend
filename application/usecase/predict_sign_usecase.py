from Domain.entities import  PredictionResult, SignSequence
from interfaces.predictor_interface import PredictorInterface

class PredictSignUseCase:

    def __init__(self, predictor: PredictorInterface):
        self.predictor = predictor


    def execute(self, sign_sequence: SignSequence) -> PredictionResult:

        preprocessed_sequence = self.predictor.preprocess(sign_sequence.sequence)

        label, confidence = self.predictor.predict(preprocessed_sequence)

        if confidence < 0.8:
            return PredictionResult("Unknown", confidence)
        
        return PredictionResult(label, confidence)
        


        