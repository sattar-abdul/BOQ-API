import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

# ============================================
# LOAD ARTIFACTS (LOCAL FILES, NO GOOGLE DRIVE)
# ============================================

# Assumes these files are in the same folder as inference.py
preprocessor = joblib.load("preprocessor.pkl")

# Load best iteration counts for each of the 33 models
best_iters = joblib.load("best_iterations.pkl")

# Load all 33 XGBoost models
models = []
for i in range(33):
    model = xgb.Booster()
    model.load_model(f"xgb_model_target_{i}.json")
    models.append(model)

# ============================
# MATERIAL LISTS & UNIT MAPPING
# ============================

material_cols = [
    'ACSR_Moose_tons','ACSR_Zebra_tons','AAAC_tons','OPGW_km','Earthwire_km',
    'Tower_Steel_MT','Angle_Tower_MT','Bolts_Nuts_pcs','Disc_Insulators_units',
    'Longrod_Insulators_units','Vibration_Dampers_pcs','Spacer_Dampers_pcs',
    'Clamp_Fittings_sets','Conductor_Accessories_sets','Earth_Rods_units',
    'Foundation_Concrete_m3','Control_Cable_m','Power_Cable_m',
    'Transformer_MVA_units','Power_Transformer_units','Circuit_Breaker_units',
    'Isolator_units','CT_PT_sets','Relay_Panels_units','Busbar_MT',
    'Cement_MT','Sand_m3','Aggregate_m3','Earthing_Mat_sets',
    'MC501_units','Cable_Trays_m','Lighting_Protection_sets','Misc_Hardware_lots'
]

material_units = {
    "ACSR_Moose_tons": "tons",
    "ACSR_Zebra_tons": "tons",
    "AAAC_tons": "tons",
    "OPGW_km": "km",
    "Earthwire_km": "km",

    "Tower_Steel_MT": "MT",
    "Angle_Tower_MT": "MT",
    "Bolts_Nuts_pcs": "pcs",
    "Disc_Insulators_units": "units",
    "Longrod_Insulators_units": "units",
    "Vibration_Dampers_pcs": "pcs",
    "Spacer_Dampers_pcs": "pcs",
    "Clamp_Fittings_sets": "sets",
    "Conductor_Accessories_sets": "sets",
    "Earth_Rods_units": "units",
    "Foundation_Concrete_m3": "m3",

    "Control_Cable_m": "m",
    "Power_Cable_m": "m",

    "Transformer_MVA_units": "units",
    "Power_Transformer_units": "units",
    "Circuit_Breaker_units": "units",
    "Isolator_units": "units",
    "CT_PT_sets": "sets",
    "Relay_Panels_units": "units",
    "Busbar_MT": "MT",

    "Cement_MT": "MT",
    "Sand_m3": "m3",
    "Aggregate_m3": "m3",

    "Earthing_Mat_sets": "sets",
    "MC501_units": "units",
    "Cable_Trays_m": "m",
    "Lighting_Protection_sets": "sets",
    "Misc_Hardware_lots": "lots"
}

# =====================
# POST-PROCESSING LOGIC
# =====================

def postprocess_boq(pred_values, project_type, project_spec=None):
    """
    Apply inverse log-transform, enforce business rules, and integer rounding.

    pred_values: list/array of 33 raw model outputs (log1p scale)
    project_type: string, e.g. "Transmission_Line", "Substation", "Line+Substation"
    project_spec: original project spec dict (used for route_km)
    """
    pred = np.array(pred_values)

    # 1) Inverse log-transform (model trained on log1p)
    pred = np.expm1(pred)

    # Helper to get index by material name
    def idx(name): 
        return material_cols.index(name)

    # 2) Zero-out substation-related items when project_type is Transmission_Line
    substation_items = [
        "Power_Cable_m","Transformer_MVA_units","Power_Transformer_units",
        "Circuit_Breaker_units","Isolator_units","CT_PT_sets",
        "Relay_Panels_units","Busbar_MT","Cable_Trays_m"
    ]

    if project_type == "Transmission_Line":
        for itm in substation_items:
            pred[idx(itm)] = 0

    # 3) Snap transformer MVA rating to one of the allowed discrete values
    valid_mva = [40, 63, 100, 160, 315]

    if project_type != "Transmission_Line":
        t_i = idx("Transformer_MVA_units")
        raw = pred[t_i]
        pred[t_i] = min(valid_mva, key=lambda x: abs(x - raw))
    else:
        pred[idx("Transformer_MVA_units")] = 0

    # 4) Constrain number of transformers between 1–3 (if not TL)
    pt_i = idx("Power_Transformer_units")
    if project_type == "Transmission_Line":
        pred[pt_i] = 0
    else:
        pred[pt_i] = int(np.clip(round(pred[pt_i]), 1, 3))

    # 5) Ensure OPGW & Earthwire >= route_km (if provided)
    if project_spec:
        route_km = float(project_spec.get("route_km", 0))
        if route_km > 0:
            for name in ["OPGW_km", "Earthwire_km"]:
                j = idx(name)
                if pred[j] < 0.9 * route_km:
                    pred[j] = route_km

    # 6) Clip negative values to 0
    pred = np.where(pred < 0, 0, pred)

    # 7) Round everything to integer quantities
    for i in range(len(pred)):
        pred[i] = int(round(pred[i]))

    return pred

# =====================
# MAIN PREDICTION API
# =====================

def predict_BOQ(project_spec):
    """
    Main function to get BOQ as a pandas DataFrame.

    project_spec = {
        "project_type": "Transmission_Line" / "Substation" / "Line+Substation",
        "state": "...",
        "voltage_kV": ...,
        "route_km": ...,
        "avg_span_m": ...,
        "tower_count": ...,
        "num_circuits": ...,
        "terrain_type": "...",
        "logistics_difficulty_score": ...,
        "substation_type": "...",
        "no_of_bays": ...,
        "project_budget_in_crores": ...
    }
    """

    # Prepare input in the same way as training
    sample = pd.DataFrame([project_spec])
    sample_prepared = preprocessor.transform(sample)
    dmatrix = xgb.DMatrix(sample_prepared)

    raw_preds = []

    # Loop through all 33 models and predict
    for i, model in enumerate(models):
        best_rounds = best_iters[i]   # apply pruning / early stopping round
        pred = model.predict(dmatrix, iteration_range=(0, best_rounds))
        raw_preds.append(pred[0])

    # Apply post-processing business logic
    final_preds = postprocess_boq(raw_preds, project_spec["project_type"], project_spec)

    # Create Final BOQ Table as DataFrame
    boq_df = pd.DataFrame({
        "Material": material_cols,
        "Predicted Quantity": final_preds,
        "Unit": [material_units[m] for m in material_cols]
    })

    return boq_df

def predict(project_spec):
    """
    Convenience wrapper for use in APIs.
    Returns list of dicts instead of DataFrame (JSON-friendly).
    """
    boq_df = predict_BOQ(project_spec)
    return boq_df.to_dict(orient="records")
