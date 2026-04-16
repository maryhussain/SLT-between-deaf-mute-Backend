from infrastructure.predictor.tensorflowlitepredictor import TensorFlowLitePredictor
from application.usecase.predict_sign_usecase import PredictSignUseCase


predictor_instance = TensorFlowLitePredictor(
    model_path='',
    scaler_path='',
    action_labels=[]
)


def get_predict_sign_usecase() -> PredictSignUseCase:
    
    return PredictSignUseCase(predictor_instance)
