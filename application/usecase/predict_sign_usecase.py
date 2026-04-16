from Domain.entities import  PredictionResult, SignSequence
from interfaces.predictor_interface import PredictorInterface

class PredictSignUseCase:

    def __init__(self, predictor: PredictorInterface):
        self.predictor = predictor


    
    def execute(self, sign_sequence: SignSequence) -> PredictionResult:

        try:

            preprocessed_sequence = self.predictor.preprocess(sign_sequence.sequence)

            label, confidence = self.predictor.predict(preprocessed_sequence)

            if confidence < 0.8:
                return PredictionResult("Unknown", confidence)
            
            return PredictionResult(label, confidence)
        
        except ValueError as e :
            print(f"Error during prediction: {e}")
            return PredictionResult("Unknown", 0.0)
        
        


        