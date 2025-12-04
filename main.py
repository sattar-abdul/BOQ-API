from fastapi import FastAPI
from pydantic import BaseModel
from inference import predict_BOQ, split_monthly_from_total, final_postprocess_monthly

app = FastAPI(title="BOQ Forecast API")

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

    # Only one extra field required
    months: int = 12


@app.get("/")
def root():
    return {"status": "BOQ API running"}


@app.post("/predict")
def predict_all(spec: ProjectSpec):

    spec_dict = spec.dict()
    months = spec_dict.pop("months")  # extract months

    # 1️⃣ TOTAL BOQ
    boq_df = predict_BOQ(spec_dict)

    # 2️⃣ MONTHLY BOQ
    monthly_raw, _curves = split_monthly_from_total(
        boq_df,
        spec_dict,
        months=months
    )

    # 3️⃣ POSTPROCESS MONTHLY
    monthly_clean = final_postprocess_monthly(monthly_raw, boq_df)

    return {
        "total_boq": boq_df.to_dict(orient="records"),
        "monthly_boq": monthly_clean.to_dict(orient="records"),
        "months": months
    }
