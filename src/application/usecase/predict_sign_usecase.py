from src.Domain.entities import  PredictionResult, SignImageSequence
from src.application.interfaces.predictor_interface import PredictorInterface
from config import Config

class PredictSignUseCase:

    def __init__(self, predictor: PredictorInterface):
        self.predictor = predictor


    def execute(self, sign_image_sequence: SignImageSequence) -> PredictionResult:

        try:

            preprocessed_data = self.predictor.preprocess(sign_image_sequence.images_bytes)

            label, confidence = self.predictor.predict(preprocessed_data)

            if confidence < Config.CONFIDENCE_THRESHOLD :
                return PredictionResult("Unknown", confidence)
            
            return PredictionResult(label, confidence)
        
        except ValueError  :
            
            return PredictionResult("Unknown", 0.0)
        
        


        