#goal of this file: Schemas.py is to faciliatate the API contract(request/response)
#So its essentially the contract/connection between the model server and outside world

from typing import Dict, Union, Optional
from pydantic import BaseModel, Field

Number = Union[int, float]

class predictRequest(BaseModel):

    #features: Dict[str, Union[Number, str]] = Field(..., description="Raw feature map with keys: sepal_length, sepal_width, petal_length, petal_width")

    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class predictIrisResponse(BaseModel):

    #setting up the output from the fastapi
    predicted_class: int
    probabilities: list[float]

    model_version: str

