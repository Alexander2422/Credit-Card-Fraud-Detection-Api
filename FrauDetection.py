from pydantic import BaseModel

class FrauDetection(BaseModel):
    distanceFromHome: float
    distanceFromLastTransaction: float
    repeatRetailer: float
    usedChip: float
    usedPinNumber: float
    onlineOrder:float