from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from predictor import predict_delay

app = FastAPI(title="Flight Delay Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class FlightInput(BaseModel):
    MONTH: int
    DAY: int
    DAY_OF_WEEK: int
    SCHEDULED_TIME: float
    AIRLINE_CODE: str
    ORIGIN_AIRPORT: str
    DESTINATION_AIRPORT: str
    DISTANCE: float
    ORIGIN_LAT: float
    ORIGIN_LON: float
    DIVERTED: int
    DEP_HOUR: int
    ARR_HOUR: int

@app.get("/")
def health():
    return {"status": "ok", "message": "Flight Delay Prediction API is running"}

@app.post("/predict")
def predict(flight: FlightInput):
    try:
        result = predict_delay(flight.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
