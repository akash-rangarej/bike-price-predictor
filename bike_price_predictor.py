# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, re

st.set_page_config(page_title="Bike Price Predictor", page_icon="🏍️")

# --- Load helpers ---
def read_cleaned_bikes(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.stop()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_excel(path)

@st.cache_data(show_spinner=False)
def load_meta(data_path: str):
    df = read_cleaned_bikes(data_path)
    locs = sorted(df["location"].dropna().astype(str).unique())
    models = sorted(df["model_name"].dropna().astype(str).unique())
    locs_map = df["location"].value_counts().to_dict()
    brand_map = df["model_name"].value_counts().to_dict()
    return locs, models, locs_map, brand_map

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    return joblib.load(model_path)

# Compatible with pickled transformers that call data_transform_pipeline(df)
def data_transform_pipeline(df: pd.DataFrame,
                            locations_map: dict | None = None,
                            brand_mean_map: dict | None = None) -> pd.DataFrame:
    if locations_map is None:
        locations_map = LOCS_MAP if "LOCS_MAP" in globals() else {}
    if brand_mean_map is None:
        brand_mean_map = BRAND_MAP if "BRAND_MAP" in globals() else {}

    def freq_encode(location):
        f = locations_map.get(location, 0)
        return "very high" if f > 1000 else "high" if f > 500 else "medium" if f > 100 else "low" if f > 10 else "very low"

    def extract_numerics(text):
        if pd.isna(text): return None
        m = re.findall(r"\d+\.?\d*", str(text))
        return float(max(m)) if m else None

    out = df.copy()
    for c in ["mileage","kms_driven","power","owner","location","model_name","model_year"]:
        if c not in out.columns: out[c] = np.nan

    if out["mileage"].dtype == "O":
        out["mileage"] = out["mileage"].astype(str).str.strip().replace({"":None}).apply(extract_numerics)
        out["mileage"] = out["mileage"].fillna(out["mileage"].median())
    if out["kms_driven"].dtype == "O":
        out["kms_drven"] = out["kms_driven"].apply(extract_numerics)
    if out["power"].dtype == "O":
        out["power"] = out["power"].apply(extract_numerics)

    owner_map = {"first owner":4,"second owner":3,"third owner":2,"fourth owner or more":1}
    out["owner"] = out["owner"].map(owner_map)

    out["location"] = out["location"].apply(freq_encode)
    freq_map = {"very high":5,"high":4,"medium":3,"low":2,"very low":1}
    out["location"] = out["location"].map(freq_map)

    if len(out["model_name"]) >= 1:
        out["engine_cc"] = out["model_name"].astype(str).str.extract(r"(\d{2,4})\s?(?:cc|CC|Cc)?").astype(float)

    out["model_name"] = out["model_name"].map(brand_mean_map)
    return out[["model_year","model_name","kms_driven","owner","location","mileage","power","engine_cc"]]

# --- Sidebar paths (change if needed) ---
st.sidebar.header("Files")
model_path = st.sidebar.text_input("Model (.pkl)", "bike_price_predictor.pkl")
data_path  = st.sidebar.text_input("Dataset (cleaned_bikes.xls)", "cleaned_bikes.xls")

# Load meta + model
try:
    LOCS, MODEL_NAMES, LOCS_MAP, BRAND_MAP = load_meta(data_path)
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# --- UI ---
st.title("🏍️ Simple Bike Price Predictor")
st.caption("Enter details and get an estimated price (₹).")

curr_year = pd.Timestamp.now().year
owners = ['first owner','second owner','third owner','fourth owner or more']

c1, c2, c3 = st.columns(3)
with c1:
    model_year = st.number_input("Model year", 1990, curr_year, 2018)
with c2:
    model_name = st.selectbox("Model name", MODEL_NAMES, index=min(10, len(MODEL_NAMES)-1) if MODEL_NAMES else 0)
with c3:
    kms_driven = st.number_input("KMs driven", 0, 300000, 30000, step=500)

c4, c5, c6 = st.columns(3)
with c4:
    owner = st.selectbox("Owner", owners)
with c5:
    location = st.selectbox("Location", LOCS, index=min(10, len(LOCS)-1) if LOCS else 0)
with c6:
    mileage = st.number_input("Mileage (kmpl)", 0.0, 150.0, 40.0, step=0.5)

c7, c8 = st.columns(2)
with c7:
    power = st.number_input("Power (bhp)", 0.0, 100.0, 12.0, step=0.5)
with c8:
    engine_cc = st.number_input("Engine CC", 50.0, 1000.0, 350.0, step=5.0)

cols = ["model_year","model_name","kms_driven","owner","location","mileage","power","engine_cc"]
raw = pd.DataFrame([[model_year, model_name, kms_driven, owner, location, mileage, power, engine_cc]], columns=cols)

if st.button("Predict price", type="primary"):
    try:
        # Try direct (if the pickle contains its own preprocessors)
        y = model.predict(raw)
        price = float(y[0])
        st.success("Predicted successfully (direct).")
    except Exception as e1:
        try:
            trans = data_transform_pipeline(raw, LOCS_MAP, BRAND_MAP)
            y = model.predict(trans)
            price = float(y[0])
            st.info("Model expected transformed features — preprocessing applied.")
        except Exception as e2:
            st.error("Prediction failed with both raw and transformed inputs.")
            st.exception(e2)
            st.stop()

    st.metric("Estimated price (₹)", f"{round(price):,}")
