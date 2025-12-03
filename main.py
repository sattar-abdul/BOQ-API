from fastapi import FastAPI
from pydantic import BaseModel
from inference import predict  # your inference loader

app = FastAPI(title="BOQ Forecast API")

# Input schema
class ProjectSpec(BaseModel):
    project_type: str
    state: str
    voltage_kV: int
    route_km: float
    avg_span_m: float
    tower_count: int
    num_circuits: int
    terrain_type: str
    logistics_difficulty_score: int
    substation_type: str
    no_of_bays: int
    project_budget_in_crores: float

@app.get("/")
def root():
    return {"status": "BOQ Forecast API running"}

@app.post("/predict")
def get_prediction(spec: ProjectSpec):
    spec_dict = spec.dict()
    result = predict(spec_dict)
    return {"prediction": result}
