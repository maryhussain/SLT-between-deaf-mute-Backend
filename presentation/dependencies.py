from src.infrastructure.predictor.tensorflowpredictor import TensorFlowPredictor
from src.application.usecase.predict_sign_usecase import PredictSignUseCase
from config import Config

predictor_instance = TensorFlowPredictor(
    model_path=Config.MODEL_PATH,
    action_labels=Config.ACTIONS
)


def get_predict_sign_usecase() -> PredictSignUseCase:
    
    return PredictSignUseCase(predictor_instance)
