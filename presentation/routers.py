from typing import List
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from src.application.usecase.predict_sign_usecase import PredictSignUseCase
from presentation.schemas import PredictionResponse
from presentation.dependencies import get_predict_sign_usecase
from src.Domain.entities import  SignImageSequence

router = APIRouter()

@router.post('/predict')
async def predict_sign(
    files : List[UploadFile] = File(...),
    usecase : PredictSignUseCase = Depends(get_predict_sign_usecase)
) -> PredictionResponse:
    
    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No files uploaded")
   
    try:
        image_bytes = [await file.read() for file in files]
        sequence_entity = SignImageSequence(image_bytes)
        prediction_result = usecase.execute(sequence_entity)

        return PredictionResponse(label=prediction_result.label, confidence=prediction_result.confidence)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="An unexpected error occurred: " + str(e))   



    

