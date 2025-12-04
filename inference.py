import numpy as np
import pandas as pd
import xgboost as xgb
import joblib

# ================================
# LOAD PREPROCESSOR + MODELS
# ================================
preprocessor = joblib.load("preprocessor.pkl")
best_iters = joblib.load("best_iterations.pkl")

models = []
for i in range(33):
    model = xgb.Booster()
    model.load_model(f"xgb_model_target_{i}.json")
    models.append(model)

# ================================
# MATERIAL LIST + UNITS
# ================================
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

# ================================
# POSTPROCESSING FOR OVERALL BOQ
# ================================
def postprocess_boq(pred_values, project_type, project_spec=None):
    pred = np.array(pred_values)

    # 1. Inverse log-transform
    pred = np.expm1(pred)

    def idx(name):
        return material_cols.index(name)

    # 2. Zero substation items for TL
    substation_items = [
        "Power_Cable_m","Transformer_MVA_units","Power_Transformer_units",
        "Circuit_Breaker_units","Isolator_units","CT_PT_sets",
        "Relay_Panels_units","Busbar_MT","Cable_Trays_m"
    ]

    if project_type == "Transmission_Line":
        for itm in substation_items:
            pred[idx(itm)] = 0

    # 3. Snap Transformer MVA
    valid_mva = [40, 63, 100, 160, 315]
    if project_type != "Transmission_Line":
        t_i = idx("Transformer_MVA_units")
        raw = pred[t_i]
        pred[t_i] = min(valid_mva, key=lambda x: abs(x - raw))
    else:
        pred[idx("Transformer_MVA_units")] = 0

    # 4. Fix number of transformers
    pt_i = idx("Power_Transformer_units")
    if project_type == "Transmission_Line":
        pred[pt_i] = 0
    else:
        pred[pt_i] = int(np.clip(round(pred[pt_i]), 1, 3))

    # 5. Ensure OPGW & Earthwire >= route_km
    if project_spec:
        route_km = float(project_spec.get("route_km", 0))
        if route_km > 0:
            for name in ["OPGW_km", "Earthwire_km"]:
                j = idx(name)
                if pred[j] < 0.9 * route_km:
                    pred[j] = route_km

    # 6. Clip negatives
    pred = np.where(pred < 0, 0, pred)

    # 7. Round to integers
    pred = np.round(pred).astype(int)

    return pred

# ================================
# PREDICT TOTAL BOQ
# ================================
def predict_BOQ(project_spec):
    df = pd.DataFrame([project_spec])
    X = preprocessor.transform(df)
    dmat = xgb.DMatrix(X)

    raw_preds = []

    for i, model in enumerate(models):
        best_rounds = best_iters[i]
        pred = model.predict(dmat, iteration_range=(0, best_rounds))
        raw_preds.append(pred[0])

    final_preds = postprocess_boq(raw_preds, project_spec["project_type"], project_spec)

    boq_df = pd.DataFrame({
        "Material": material_cols,
        "Predicted Quantity": final_preds,
        "Unit": [material_units[m] for m in material_cols]
    })

    return boq_df

# ================================
# MATERIAL CATEGORY FOR MONTHLY BOQ
# ================================
material_category = {
    'ACSR_Moose_tons':'conductor','ACSR_Zebra_tons':'conductor','AAAC_tons':'conductor',
    'OPGW_km':'conductor','Earthwire_km':'conductor',
    'Tower_Steel_MT':'tower','Angle_Tower_MT':'tower','Bolts_Nuts_pcs':'tower',
    'Disc_Insulators_units':'tower','Longrod_Insulators_units':'tower',
    'Vibration_Dampers_pcs':'tower','Spacer_Dampers_pcs':'tower',
    'Clamp_Fittings_sets':'tower','Conductor_Accessories_sets':'tower',
    'Earth_Rods_units':'tower','Foundation_Concrete_m3':'civil',
    'Control_Cable_m':'cabling','Power_Cable_m':'cabling',
    'Transformer_MVA_units':'equipment','Power_Transformer_units':'equipment',
    'Circuit_Breaker_units':'equipment','Isolator_units':'equipment',
    'CT_PT_sets':'equipment','Relay_Panels_units':'equipment','Busbar_MT':'equipment',
    'Cement_MT':'civil','Sand_m3':'civil','Aggregate_m3':'civil',
    'Earthing_Mat_sets':'civil','MC501_units':'civil',
    'Cable_Trays_m':'cabling','Lighting_Protection_sets':'misc','Misc_Hardware_lots':'misc'
}

# ================================
# MONTHLY CURVES (Your PGCIL templates)
# ================================
base_curves_12 = {
    'conductor': np.array([0.005,0.01,0.03,0.07,0.12,0.15,0.20,0.18,0.12,0.06,0.01,0.00]),
    'tower':     np.array([0.03,0.06,0.12,0.16,0.18,0.16,0.12,0.08,0.05,0.02,0.01,0.01]),
    'civil':     np.array([0.25,0.22,0.18,0.12,0.08,0.06,0.04,0.02,0.01,0.01,0.00,0.01]),
    'equipment': np.array([0.00,0.01,0.02,0.05,0.12,0.20,0.20,0.18,0.12,0.06,0.03,0.01]),
    'cabling':   np.array([0.00,0.00,0.02,0.05,0.10,0.15,0.22,0.20,0.12,0.08,0.04,0.02]),
    'misc':      np.array([0.01,0.02,0.04,0.07,0.12,0.18,0.18,0.15,0.12,0.06,0.03,0.02])
}
for k,v in base_curves_12.items():
    base_curves_12[k] = v / v.sum()

# ================================
# MONTHLY CURVE TRANSFORMATIONS
# ================================
def stretch_curve(curve12, months):
    if months == 12:
        out = curve12.copy()
    else:
        x_old = np.linspace(0,1,12)
        x_new = np.linspace(0,1,months)
        out = np.interp(x_new, x_old, curve12)
    out = np.maximum(out, 0)
    s = out.sum()
    return out / s if s > 0 else np.ones(months)/months

def shift_curve(curve, shift_months):
    months = len(curve)
    x = np.arange(months)/(months-1)
    x_shifted = np.clip(x - (shift_months/months), 0, 1)
    new = np.interp(x, x_shifted, curve)
    return new / new.sum()

def flatten_curve(curve, factor):
    months = len(curve)
    uniform = np.ones(months)/months
    out = (1-factor)*curve + factor*uniform
    return out / out.sum()

# ================================
# MONTHLY BOQ GENERATION
# ================================
def split_monthly_from_total(boq_df, project_spec, months=12):
    monsoon_risk = float(project_spec.get("monsoon_risk", 0))
    contractor_speed = float(project_spec.get("contractor_speed", 1.0))
    terrain = project_spec.get("terrain_type","plain")

    # shifts
    monsoon_delay = min(2.0, monsoon_risk*3.0) if monsoon_risk > 0.3 else 0
    speed_shift = -0.5 * (contractor_speed - 1.0)
    terrain_delay = 0.8 if terrain == "hilly" else (0.5 if terrain == "coastal" else 0)

    shift_months = monsoon_delay + terrain_delay + speed_shift

    # curves
    stretched = {}
    for cat, c12 in base_curves_12.items():
        c = stretch_curve(c12, months)
        if shift_months != 0:
            c = shift_curve(c, shift_months)
        if contractor_speed < 0.8:
            c = flatten_curve(c, min(0.5, 0.8 - contractor_speed))
        stretched[cat] = c

    materials = list(boq_df["Material"])
    qtys = list(boq_df["Predicted Quantity"])

    monthly = []
    for m in range(months):
        monthly.append({"Month": m+1})

    for mat, total in zip(materials, qtys):
        cat = material_category.get(mat, "misc")
        curve = stretched.get(cat, stretched["misc"])
        per_month = curve * float(total)

        for m in range(months):
            monthly[m][mat] = per_month[m]

    monthly_df = pd.DataFrame(monthly)

    # integer rounding with conservation
    for mat, total in zip(materials, qtys):
        arr = monthly_df[mat].values.astype(float)
        if total == 0:
            monthly_df[mat] = 0
            continue

        rounded = np.floor(arr).astype(int)
        residue = int(round(total - rounded.sum()))

        fracs = arr - np.floor(arr)
        idxs = np.argsort(-fracs)

        i = 0
        while residue > 0 and i < len(idxs):
            rounded[idxs[i]] += 1
            residue -= 1
            i += 1

        i = 0
        while residue < 0 and i < len(idxs):
            j = idxs[::-1][i]
            if rounded[j] > 0:
                rounded[j] -= 1
                residue += 1
            i += 1

        monthly_df[mat] = rounded

    return monthly_df, stretched

# ================================
# FINAL MONTHLY POSTPROCESSING
# ================================
def final_postprocess_monthly(monthly_df, boq_df):
    df = monthly_df.copy()

    materials = list(boq_df["Material"])
    total_qty = boq_df.set_index("Material")["Predicted Quantity"].to_dict()

    # 1. Smooth
    for mat in materials:
        df[mat] = df[mat].rolling(3, center=True, min_periods=1).mean()

    # 2. Clip negatives
    df[materials] = df[materials].clip(lower=0)

    # 3. Tail fix
    for mat in materials:
        tot = total_qty[mat]
        if tot == 0:
            continue

        tail = df.loc[df.index[-1], mat]
        peak = df[mat].max()

        if peak > 0 and tail < 0.05 * peak:
            df.loc[df.index[-2], mat] += tail
            df.loc[df.index[-1], mat] = 0

    # 4. Conserve totals
    for mat in materials:
        desired = total_qty[mat]
        current = df[mat].sum()
        diff = round(desired - current)

        if diff > 0:
            idx = df[mat].idxmax()
            df.loc[idx, mat] += diff

        elif diff < 0:
            diff = abs(diff)
            for idx in df[mat].sort_values(ascending=False).index:
                if diff == 0:
                    break
                available = df.loc[idx, mat]
                remove = min(available, diff)
                df.loc[idx, mat] -= remove
                diff -= remove

        df[mat] = df[mat].clip(lower=0)

    # 5. Final rounding
    df[materials] = df[materials].round().astype(int)

    # 6. Conservation check
    for mat in materials:
        desired = total_qty[mat]
        current = df[mat].sum()
        diff = desired - current

        if diff > 0:
            idx = df[mat].idxmax()
            df.loc[idx, mat] += diff

        elif diff < 0:
            diff = abs(diff)
            for idx in df[mat].sort_values(ascending=False).index:
                if diff == 0:
                    break
                available = df.loc[idx, mat]
                remove = min(available, diff)
                df.loc[idx, mat] -= remove
                diff -= remove

        df[mat] = df[mat].clip(lower=0)

    return df
