"""
AI House Price Prediction — Streamlit App
==========================================

WHAT THIS APP DOES (for presentation):
This app loads the trained Random Forest pipeline (models/house_price_model.pkl)
that was built and saved in the project notebook. That pipeline already contains
BOTH the preprocessing steps (imputing missing values, scaling numeric columns,
one-hot encoding categorical columns) AND the trained Random Forest model.

Because the pipeline was trained on 77 raw feature columns (from the Ames
Housing dataset), it expects a full row of 77 columns to make a prediction —
not just the few features a user would realistically want to type in.

To keep the app SIMPLE, we ask the user for only the ~12-16 features that a
non-technical person understands and that matter most (see
models/feature_importance.csv). Every other required column is filled in
automatically with a sensible default — the exact median (for numeric
columns) or most-frequent value (for categorical columns) that the model's
own preprocessing pipeline learned during training. These defaults were
extracted directly from the fitted pipeline, so they are guaranteed to match
how the model was actually trained (see the SimpleImputer `statistics_`
inside the pipeline's "preprocessor" step).

NOTE: The saved model (models/house_price_model.pkl) is only ever loaded
here — it is never modified or retrained by this app.
"""

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# CONFIG / CONSTANTS
# ---------------------------------------------------------------------------

MODEL_PATH = "models/house_price_model.pkl"

# The model predicts in the ORIGINAL dataset's price scale (US Dollars,
# Ames, Iowa housing data). We only convert to INR for display purposes,
# using a fixed, clearly-labelled constant — this does NOT change the model.
USD_TO_INR = 85

# ---------------------------------------------------------------------------
# DEFAULT VALUES FOR ALL "BACKGROUND" FEATURES
# ---------------------------------------------------------------------------
# These are the EXACT median (numeric) / most-frequent (categorical) values
# that the saved pipeline's own imputers learned from the training data.
# They were extracted directly from the fitted pipeline
# (preprocessor -> num/cat -> imputer -> statistics_), so a user who accepts
# all defaults gets a prediction consistent with a "typical" training-set house.

NUMERIC_DEFAULTS = {
    "MSSubClass": 50.0, "LotFrontage": 69.0, "LotArea": 9600.0,
    "OverallQual": 6.0, "OverallCond": 5.0, "YearBuilt": 1973.0,
    "YearRemodAdd": 1994.0, "MasVnrArea": 0.0, "BsmtFinSF1": 395.5,
    "BsmtFinSF2": 0.0, "BsmtUnfSF": 459.5, "TotalBsmtSF": 1000.0,
    "1stFlrSF": 1090.5, "2ndFlrSF": 0.0, "LowQualFinSF": 0.0,
    "GrLivArea": 1466.0, "BsmtFullBath": 0.0, "BsmtHalfBath": 0.0,
    "FullBath": 2.0, "HalfBath": 0.0, "BedroomAbvGr": 3.0,
    "KitchenAbvGr": 1.0, "TotRmsAbvGrd": 6.0, "Fireplaces": 1.0,
    "GarageYrBlt": 1977.0, "GarageCars": 2.0, "GarageArea": 480.0,
    "WoodDeckSF": 0.0, "OpenPorchSF": 27.0, "EnclosedPorch": 0.0,
    "3SsnPorch": 0.0, "ScreenPorch": 0.0, "PoolArea": 0.0,
    "MiscVal": 0.0, "MoSold": 6.0, "YrSold": 2008.0,
}

CATEGORICAL_DEFAULTS = {
    "MSZoning": "RL", "Street": "Pave", "Alley": "None", "LotShape": "Reg",
    "LandContour": "Lvl", "Utilities": "AllPub", "LotConfig": "Inside",
    "LandSlope": "Gtl", "Neighborhood": "NAmes", "Condition1": "Norm",
    "Condition2": "Norm", "BldgType": "1Fam", "HouseStyle": "1Story",
    "RoofStyle": "Gable", "RoofMatl": "CompShg", "Exterior1st": "VinylSd",
    "Exterior2nd": "VinylSd", "MasVnrType": "None", "ExterQual": "TA",
    "ExterCond": "TA", "Foundation": "PConc", "BsmtQual": "TA",
    "BsmtCond": "TA", "BsmtExposure": "No", "BsmtFinType1": "Unf",
    "BsmtFinType2": "Unf", "Heating": "GasA", "HeatingQC": "Ex",
    "CentralAir": "Y", "Electrical": "SBrkr", "KitchenQual": "TA",
    "Functional": "Typ", "FireplaceQu": "None", "GarageType": "Attchd",
    "GarageFinish": "Unf", "GarageQual": "TA", "GarageCond": "TA",
    "PavedDrive": "Y", "PoolQC": "None", "Fence": "None",
    "MiscFeature": "None", "SaleType": "WD", "SaleCondition": "Normal",
}

# Columns where the notebook manually filled missing values with the string
# "None" (meaning "feature does not exist", e.g. no pool / no fence) rather
# than the pipeline's own "most frequent" imputation. We keep "None" as
# their default here so unseen/blank cases stay consistent with training.
STRUCTURAL_NONE_COLS = [
    "PoolQC", "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageType",
    "GarageFinish", "GarageQual", "GarageCond", "BsmtQual", "BsmtCond",
    "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "MasVnrType",
]

ALL_DEFAULTS = {**NUMERIC_DEFAULTS, **CATEGORICAL_DEFAULTS}

# Valid dropdown choices for the categorical fields we expose to the user,
# taken from the categories the OneHotEncoder actually learned during
# training (so we never send the model a category it has never seen).
NEIGHBORHOOD_CHOICES = [
    "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr",
    "Crawfor", "Edwards", "Gilbert", "IDOTRR", "MeadowV", "Mitchel",
    "NAmes", "NPkVill", "NWAmes", "NoRidge", "NridgHt", "OldTown",
    "SWISU", "Sawyer", "SawyerW", "Somerst", "StoneBr", "Timber", "Veenker",
]
QUALITY_CHOICES = ["Ex", "Gd", "TA", "Fa", "Po"]  # Excellent..Poor

# ---------------------------------------------------------------------------
# LOAD THE SAVED PIPELINE (cached so it only loads once per session)
# ---------------------------------------------------------------------------


@st.cache_resource
def load_model():
    """Load the existing trained pipeline. Never retrains or modifies it."""
    return joblib.load(MODEL_PATH)


model = load_model()

# ---------------------------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------------------------

st.set_page_config(page_title="AI House Price Prediction", page_icon="🏠")

st.title("🏠 AI House Price Prediction")
st.write(
    "Enter the key characteristics of a house below. The model was trained "
    "on the Ames Housing dataset using a Random Forest pipeline. You only "
    "need to fill in the fields below — every other feature the model "
    "requires is automatically filled with a **typical / default value** "
    "learned from the training data, so you don't have to fill 77 fields "
    "by hand."
)

# ---------------------------------------------------------------------------
# INPUT FORM — only the most important, human-understandable features
# ---------------------------------------------------------------------------

with st.form("prediction_form"):

    st.subheader("Size & Layout")
    col1, col2 = st.columns(2)
    with col1:
        gr_liv_area = st.number_input(
            "Above-ground living area (sq ft)", min_value=100, max_value=10000,
            value=1500, step=50,
        )
        total_bsmt_sf = st.number_input(
            "Total basement area (sq ft)", min_value=0, max_value=6000,
            value=1000, step=50,
        )
        first_flr_sf = st.number_input(
            "1st floor area (sq ft)", min_value=100, max_value=6000,
            value=1100, step=50,
        )
        bsmt_fin_sf1 = st.number_input(
            "Finished basement area (sq ft)", min_value=0, max_value=6000,
            value=400, step=50,
        )
    with col2:
        lot_area = st.number_input(
            "Lot area (sq ft)", min_value=500, max_value=100000,
            value=9600, step=100,
        )
        tot_rms_abv_grd = st.number_input(
            "Total rooms above ground", min_value=1, max_value=20, value=6,
        )
        full_bath = st.number_input(
            "Full bathrooms", min_value=0, max_value=6, value=2,
        )

    st.subheader("Quality & Age")
    col3, col4 = st.columns(2)
    with col3:
        overall_qual = st.slider(
            "Overall material & finish quality (1 = worst, 10 = best)",
            min_value=1, max_value=10, value=6,
        )
        year_built = st.number_input(
            "Year built", min_value=1870, max_value=2026, value=1973,
        )
    with col4:
        year_remod_add = st.number_input(
            "Year remodeled (same as year built if never remodeled)",
            min_value=1870, max_value=2026, value=1994,
        )
        kitchen_qual = st.selectbox(
            "Kitchen quality", QUALITY_CHOICES,
            index=QUALITY_CHOICES.index("TA"),
        )

    st.subheader("Garage & Location")
    col5, col6 = st.columns(2)
    with col5:
        garage_cars = st.number_input(
            "Garage capacity (number of cars)", min_value=0, max_value=6,
            value=2,
        )
        garage_area = st.number_input(
            "Garage area (sq ft)", min_value=0, max_value=2000, value=480,
            step=20,
        )
    with col6:
        neighborhood = st.selectbox(
            "Neighborhood", NEIGHBORHOOD_CHOICES,
            index=NEIGHBORHOOD_CHOICES.index("NAmes"),
        )
        ext_qual = st.selectbox(
            "Exterior quality", QUALITY_CHOICES,
            index=QUALITY_CHOICES.index("TA"),
        )

    submitted = st.form_submit_button("🔮 Predict House Price")

# ---------------------------------------------------------------------------
# VALIDATION + PREDICTION
# ---------------------------------------------------------------------------

if submitted:
    # --- Basic sanity checks beyond the min/max already enforced by the
    #     number inputs above -----------------------------------------------
    errors = []
    if year_remod_add < year_built:
        errors.append("Remodel year cannot be earlier than the year built.")
    if bsmt_fin_sf1 > total_bsmt_sf:
        errors.append("Finished basement area cannot exceed total basement area.")
    if first_flr_sf > gr_liv_area:
        errors.append(
            "1st floor area cannot exceed total above-ground living area."
        )

    if errors:
        for e in errors:
            st.error(e)
    else:
        # --- Build the full 77-column input row ---------------------------
        # Start from the training-data defaults, then overwrite only the
        # fields the user actually filled in.
        row = ALL_DEFAULTS.copy()
        row.update({
            "GrLivArea": gr_liv_area,
            "TotalBsmtSF": total_bsmt_sf,
            "1stFlrSF": first_flr_sf,
            "BsmtFinSF1": bsmt_fin_sf1,
            "LotArea": lot_area,
            "TotRmsAbvGrd": tot_rms_abv_grd,
            "FullBath": full_bath,
            "OverallQual": overall_qual,
            "YearBuilt": year_built,
            "YearRemodAdd": year_remod_add,
            "KitchenQual": kitchen_qual,
            "GarageCars": garage_cars,
            "GarageArea": garage_area,
            "Neighborhood": neighborhood,
            "ExterQual": ext_qual,
        })

        # Structural "None" columns stay at their "None" default unless a
        # future version of the app exposes them — kept explicit here for
        # clarity during a presentation / code walkthrough.
        for col in STRUCTURAL_NONE_COLS:
            row.setdefault(col, "None")

        input_df = pd.DataFrame([row])

        # --- Predict --------------------------------------------------------
        # The pipeline handles imputing/scaling/encoding internally, so we
        # just pass the raw DataFrame straight in.
        predicted_usd = model.predict(input_df)[0]
        predicted_inr = predicted_usd * USD_TO_INR

        # --- Display ----------------------------------------------------
        st.success("Prediction complete!")
        st.metric("Estimated House Price (₹ INR)", f"₹ {predicted_inr:,.0f}")

        st.caption(
            f"Presentation conversion only: ₹ = USD × {USD_TO_INR} "
            "(fixed constant, not a live exchange rate)."
        )

        with st.expander("Show original model output (USD, unconverted)"):
            st.write(
                "The model itself was trained on the original dataset's "
                "price scale (US Dollars) and was not changed in any way:"
            )
            st.write(f"**Model output: $ {predicted_usd:,.2f} USD**")
