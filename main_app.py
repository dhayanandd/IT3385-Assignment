import os
import numpy as np
import pandas as pd
import streamlit as st

# PyCaret imports (regression & classification)
from pycaret.regression import load_model as load_reg_model, predict_model as predict_reg
from pycaret.classification import load_model as load_clf_model, predict_model as predict_clf

st.set_page_config(page_title="Unified ML App", page_icon="🤖", layout="centered")

@st.cache_resource(show_spinner=False)
def load_car_price_model():
    return load_reg_model("CarPricePredictor/my_pipeline_ck")

@st.cache_resource(show_spinner=False)
def load_house_price_model():
    return load_reg_model("HousingPricePredictor/melbourne_price_pipeline")

@st.cache_resource(show_spinner=False)
def load_wheat_classifier_model():
    return load_clf_model("WheatAppClassifier/wheat_classifier")

# Sidebar navigation
st.sidebar.title("IT3385 Assignment")
page = st.sidebar.radio(
    "Choose an app:",
    ["Used Car Price Predictor", "Melbourne House Price Predictor", "Wheat Type Classifier"],
    label_visibility="collapsed"
)


#  CAR PRICE PREDICTOR
if page.startswith("🚗"):
    st.header("Predict Used Car Price (INR / Lakh)")
    st.write(
        "This web app predicts the price of a used car. "
        "Please note the **Year** must be between **1998–2019** (matches training data)."
    )

    # Load model once
    car_model = load_car_price_model()

    st.subheader("Prediction Mode")
    mode = st.selectbox("", ["Single", "Batch"], label_visibility="collapsed")

    #  Single
    if mode == "Single":
        st.subheader("Single Prediction")
        st.caption("Fill in the car details, then click Predict.")

        col1, col2 = st.columns(2)
        with col1:
            location = st.text_input("Location")
            year = st.number_input("Year (1998–2019)", min_value=1998, max_value=2019, value=2015, step=1)
            kilometers_driven = st.number_input("Kilometers Driven (KM)", min_value=0, value=40000, step=100)
            fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Electric", "CNG", "LPG"])
            transmission = st.selectbox("Transmission", ["Automatic", "Manual"])
        with col2:
            owner_type = st.selectbox("Owner Type", ["First", "Second", "Third", "Fourth & Above"])
            mileage_input = st.text_input("Mileage (kmpl or km/kg)", value="18.0")
            engine = st.number_input("Engine (cc)", min_value=0, value=1197, step=10)
            power_input = st.text_input("Power (bhp)", value="82.0")
            seats = st.number_input("Seats (1–10)", min_value=1, max_value=10, value=5, step=1)

        # Parse floats safely
        def _to_float(x):
            try:
                return float(x)
            except Exception:
                return None

        mileage = _to_float(mileage_input)
        power = _to_float(power_input)

        st.markdown(
            """
            <style>
            div.stButton > button:first-child {
                background-color: #2563EB; color: white; font-size: 16px;
                border-radius: 10px; height: 3em; width: 100%;
            }
            div.stButton > button:hover { background-color: #1D4ED8; color: white; }
            </style>
            """,
            unsafe_allow_html=True
        )

        if st.button("🚗 Predict Price"):
            if not location.strip():
                st.error("Please enter a location.")
                st.stop()
            if mileage is None:
                st.error("Mileage must be a number (e.g., 18.0).")
                st.stop()
            if power is None:
                st.error("Power must be a number (e.g., 82.0).")
                st.stop()

            X = pd.DataFrame([{
                "Location": location.strip(),
                "Year": int(year),
                "Kilometers_Driven": int(kilometers_driven),
                "Fuel_Type": fuel_type,
                "Transmission": transmission,
                "Owner_Type": owner_type,
                "Mileage": mileage,
                "Engine": int(engine),
                "Power": power,
                "Seats": int(seats)
            }])

            try:
                preds = predict_reg(car_model, data=X)
                # Find prediction column (PyCaret adds columns)
                added = [c for c in preds.columns if c not in X.columns]
                prefer = ["Label", "prediction_label", "Prediction", "Predicted", "Score"]
                pred_col = next((c for c in prefer if c in preds.columns), added[0] if added else None)
                if pred_col is None:
                    raise ValueError(f"No prediction column found. Columns: {list(preds.columns)}")
                price = float(preds[pred_col].iloc[0])
                st.success(f"Predicted Price: **{price:.2f} Lakh (INR)**")
                with st.expander("See input & output (debug)"):
                    st.dataframe(preds, use_container_width=True)
            except Exception as e:
                st.error(f"Error while predicting: {e}")

    # ---------------- Batch
    else:
        st.subheader("Batch Prediction")
        st.write("Upload a CSV/XLSX with the following columns:")
        expected = [
            "Location", "Year", "Kilometers_Driven", "Fuel_Type",
            "Transmission", "Owner_Type", "Mileage", "Engine", "Power", "Seats"
        ]
        st.code(", ".join(expected))

        file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

        # Clear state if needed
        if file is None:
            st.session_state.pop("car_df_results", None)
        else:
            try:
                ext = file.name.split(".")[-1].lower()
                if ext == "csv":
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)
                st.write(f"Total records: {len(df)}")
                st.dataframe(df.head())

                missing = [c for c in expected if c not in df.columns]
                if missing:
                    st.error(f"Missing required columns: {missing}")
                else:
                    # quick validations
                    issues = []
                    invalid_years = df[(df["Year"] < 1998) | (df["Year"] > 2019)]
                    if not invalid_years.empty:
                        issues.append(f"{len(invalid_years)} rows have invalid Year (1998–2019).")
                    invalid_seats = df[(df["Seats"] < 1) | (df["Seats"] > 10)]
                    if not invalid_seats.empty:
                        issues.append(f"{len(invalid_seats)} rows have invalid Seats (1–10).")
                    if df[expected].isnull().sum().sum() > 0:
                        issues.append("Some required fields are missing (NaNs).")

                    if issues:
                        st.warning("Validation issues:\n- " + "\n- ".join(issues))
                    if st.button("🚗 Predict All Prices"):
                        if issues:
                            st.error("Fix validation issues before predicting.")
                        else:
                            with st.spinner("Predicting..."):
                                preds = predict_reg(car_model, data=df[expected])
                                # locate prediction column
                                added = [c for c in preds.columns if c not in df.columns]
                                prefer = ["Label", "prediction_label", "Prediction", "Predicted", "Score"]
                                pred_col = next((c for c in prefer if c in preds.columns), added[0] if added else None)
                                if pred_col is None:
                                    raise ValueError("No prediction column produced.")
                                out = df.copy()
                                out["Predicted_Price_Lakh"] = preds[pred_col].round(2).values
                                st.session_state["car_df_results"] = out

            except Exception as e:
                st.error(f"Error reading file: {e}")

        if "car_df_results" in st.session_state:
            st.subheader("Prediction Results")
            out = st.session_state["car_df_results"]
            st.dataframe(out, use_container_width=True)
            st.markdown("#### Summary")
            col1, col2, col3 = st.columns(3)
            vals = out["Predicted_Price_Lakh"].values
            with col1: st.metric("Average (Lakh)", f"{np.mean(vals):.2f}")
            with col2: st.metric("Min (Lakh)", f"{np.min(vals):.2f}")
            with col3: st.metric("Max (Lakh)", f"{np.max(vals):.2f}")

            csv = out.to_csv(index=False)
            st.download_button(
                "Download CSV",
                data=csv,
                file_name="car_price_predictions.csv",
                mime="text/csv"
            )


# 🏠 MELBOURNE HOUSE PRICE PREDICTOR
elif page.startswith("🏠"):
    st.header("Melbourne House Price Predictor")
    st.caption("Enter details on the left and click Predict.")

    house_model = load_house_price_model()

    st.sidebar.header("Property Inputs")
    Rooms = st.sidebar.number_input("Rooms", min_value=0, value=3, step=1)
    Bedroom2 = st.sidebar.number_input("Bedroom2 (scraped)", min_value=0, value=3, step=1)
    Bathroom = st.sidebar.number_input("Bathroom", min_value=0, value=2, step=1)
    Car = st.sidebar.number_input("Car Spaces", min_value=0, value=1, step=1)

    Distance = st.sidebar.number_input("Distance to CBD (km)", min_value=0.0, value=10.0, step=0.1)
    Landsize = st.sidebar.number_input("Landsize (sqm)", min_value=0.0, value=450.0, step=10.0)
    BuildingArea = st.sidebar.number_input("Building Area (sqm)", min_value=0.0, value=120.0, step=5.0)

    YearBuilt = st.sidebar.number_input("Year Built", min_value=1800, max_value=2025, value=1998, step=1)
    Propertycount = st.sidebar.number_input("Propertycount (in suburb)", min_value=0, value=6000, step=50)

    SaleYear = st.sidebar.number_input("Sale Year", min_value=2007, max_value=2025, value=2017, step=1)
    SaleMonth = st.sidebar.slider("Sale Month", min_value=1, max_value=12, value=6)

    Method = st.sidebar.selectbox("Sale Method", options=["S", "Other"], index=0)
    Type = st.sidebar.selectbox("Property Type", options=['h', 'u', 't', 'dev site', 'o res'], index=0)
    Region = st.sidebar.text_input("Region (or 'Other')", value="Other")
    CouncilArea = st.sidebar.text_input("Council Area (or 'Other')", value="Other")
    Suburb = st.sidebar.text_input("Suburb (or 'Other')", value="Other")

    use_location = st.sidebar.checkbox("Provide Latitude/Longitude?", value=False)
    if use_location:
        Latitude = st.sidebar.number_input("Latitude", value=-37.80, step=0.01, format="%.5f")
        Longitude = st.sidebar.number_input("Longitude", value=145.00, step=0.01, format="%.5f")
    else:
        Latitude = -37.80
        Longitude = 145.00

    def build_house_row():
        property_age = max(0, int(SaleYear) - int(YearBuilt)) if YearBuilt > 0 else 0
        return pd.DataFrame([{
            'Rooms': int(Rooms),
            'Bedroom2': int(Bedroom2),
            'Bathroom': int(Bathroom),
            'Car': int(Car),
            'Distance': float(Distance),
            'Landsize': float(Landsize),
            'BuildingArea': float(BuildingArea),
            'YearBuilt': float(YearBuilt),
            'CouncilArea': CouncilArea.strip() or 'Other',
            'Region': Region.strip() or 'Other',
            'Suburb': Suburb.strip() or 'Other',
            'Method': Method,
            'Type': Type,
            'Propertycount': int(Propertycount),
            'SaleYear': int(SaleYear),
            'SaleMonth': int(SaleMonth),
            'PropertyAge': int(property_age),
            'Latitude': float(Latitude),
            'Longitude': float(Longitude),

            # placeholders to match training schema (safe)
            'Price_per_sqm': np.nan,
            'Address': 'Unknown',
            'Seller': 'Other',
            'Postcode': '3000',
            'Date': '2017-06-15',
            'LogPrice': 0.0
        }])

    st.subheader("Enter details, then click Predict")
    if st.button("Predict Price", type="primary", use_container_width=True):
        X = build_house_row()
        try:
            preds = predict_reg(house_model, data=X)
            added = [c for c in preds.columns if c not in X.columns]
            prefer = ['Label', 'prediction_label', 'Prediction', 'Predicted', 'Score']
            pred_col = next((c for c in prefer if c in preds.columns), added[0] if added else None)
            if pred_col is None:
                raise ValueError(f"No prediction column. Columns: {list(preds.columns)}")
            price = float(preds[pred_col].iloc[0])
            st.success(f"**Predicted Price:** ${price:,.0f}")
            with st.expander("See input & output (debug)"):
                st.write("Prediction column:", pred_col)
                st.dataframe(preds, use_container_width=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")


# 🌾 WHEAT CLASSIFIER
else:
    st.header("Wheat Type Classification")

    wheat_model = load_wheat_classifier_model()

    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input("Area", min_value=0.0, value=15.0)
        perimeter = st.number_input("Perimeter", min_value=0.0, value=14.0)
        compactness = st.number_input("Compactness", min_value=0.0, value=0.87)
    with col2:
        asymmetry_coeff = st.number_input("Asymmetry Coefficient", min_value=0.0, value=2.0)
        groove = st.number_input("Groove", min_value=0.0, value=5.2)

    if st.button("Predict Wheat Type", type="primary"):
        X = pd.DataFrame([{
            'Area': area,
            'Perimeter': perimeter,
            'Compactness': compactness,
            'AsymmetryCoeff': asymmetry_coeff,
            'Groove': groove
        }])
        try:
            preds = predict_clf(wheat_model, data=X)
            # PyCaret classification usually adds 'prediction_label'
            label_col = 'prediction_label' if 'prediction_label' in preds.columns else 'Label'
            pred_class = preds.loc[0, label_col]
            st.success(f"Predicted Wheat Type: **{pred_class}**")
            with st.expander("See input & output (debug)"):
                st.dataframe(preds, use_container_width=True)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# Footer
st.markdown("---")
st.caption("Unified app • PyCaret pipelines • Streamlit")
