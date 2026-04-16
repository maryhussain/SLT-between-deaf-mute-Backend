from pydantic import BaseModel , Field, validator
from typing import List

class SignSequenceRequest(BaseModel):
    # Assuming each frame has 40 frames and each frame has 1530 features (510keypoints * 3 coordinates)
    sequence : List[List[float]] = Field(..., min_items=40 , max_items=40) 

    @validator('sequence')
    def validate_sequence(cls, v):    
        for frame_idx, feature in enumerate(v):
            if len(feature) != 1530 :
                raise ValueError(f"Frame at index {frame_idx} does not have exactly 1530 features")
            return v

class PredictionResponse(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


