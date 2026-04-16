from infrastructure.predictor.tensorflowlitepredictor import TensorFlowLitePredictor
from application.usecase.predict_sign_usecase import PredictSignUseCase
from config import Config

predictor_instance = TensorFlowLitePredictor(
    model_path=Config.MODEL_PATH,
    scaler_path=Config.SCALER_PATH,
    action_labels=Config.ACTIONS
)


def get_predict_sign_usecase() -> PredictSignUseCase:
    
    return PredictSignUseCase(predictor_instance)
