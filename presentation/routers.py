from fastapi import APIRouter, Depends, HTTPException
from presentation.schemas import SignSequenceRequest, PredictionResponse
from presentation.dependencies import get_predict_sign_usecase
from src.Domain.entities import SignSequence

router = APIRouter()

@router.post('/predict')
async def predict_sign(
    request: SignSequenceRequest,
    usecase = Depends(get_predict_sign_usecase)
) -> PredictionResponse:
   
    try:
        sequence_entity = SignSequence(sequence=request.sequence)
        prediction_result = usecase.execute(sequence_entity)

        return PredictionResponse(label=prediction_result.label, confidence=prediction_result.confidence)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred during prediction")   



    

