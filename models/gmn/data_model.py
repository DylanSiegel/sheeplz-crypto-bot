from pydantic import BaseModel, validator
from typing import Optional

class DataModel(BaseModel):
    c: float  # Close price
    v: Optional[float] = 0.0  # Volume

    @validator('c', 'v', pre=True)
    def validate_numeric(cls, v):
        if isinstance(v, (int, float)):
            return float(v)
        try:
            return float(v)
        except (ValueError, TypeError):
            raise ValueError(f"Value '{v}' is not a valid float.")