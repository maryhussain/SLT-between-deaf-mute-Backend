from pydantic import BaseModel , Field, validator
from typing import List
from config import Config

class SignSequenceRequest(BaseModel):
    # Assuming each frame has 40 frames and each frame has 1530 features (510keypoints * 3 coordinates)
    sequence : List[List[float]] #= Field(..., min_items=40 , max_items=40) 

    @validator('sequence')
    def validate_sequence(cls, v):    
        fixed_v = []
        for frame in v:
            if len(frame) < Config.NUM_FEATURES: 
                padded_frame = frame + [0.0] * (Config.NUM_FEATURES - len(frame))
                fixed_v.append(padded_frame)
            elif len(frame) > Config.NUM_FEATURES:
                fixed_v.append(frame[:Config.NUM_FEATURES])
            else:
                fixed_v.append(frame)
        return fixed_v

class PredictionResponse(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


