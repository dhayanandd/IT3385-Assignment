# App to predict used car prices using a trained ML pipeline with Streamlit
import streamlit as st
import joblib
import pandas as pd

# Load trained pipeline
@st.cache_resource
def load_model():
    return joblib.load("my_pipeline_ck.pkl")

model = load_model()

st.header("Predict Used Car Price (In INR/Lakh)")
st.write("This web application predicts the price of a used car based on various features listed below. Please note that you can only select the years between 1998 and 2019 for the year or edition of the car model as the training data only includes year in that range in order for the prediction to be as accurate as possible.\n")

# Add dropdown box to choose real-time prediction request type (single or batch)
st.subheader("Please Choose Prediction Request Type")

prediction_type = st.selectbox(
    "",
    ("Single", "Batch"),
    label_visibility="collapsed"  # Hides the empty label without affecting layout
)

# Single Prediction
if prediction_type == "Single":
    st.subheader("Single Prediction")
    st.subheader("Please Insert the Details of the Car Below:")

    # Input fields for all features except Brand_Model
    location = st.text_input("Location (The location in which the car is being sold or is available for purchase)")
    year = st.number_input("Year (The year or edition of the model) (Please select between 1998 and 2019)", min_value=1998, max_value=2019, step=1)
    kilometers_driven = st.number_input("Kilometers Driven (The total kilometres driven in the car by the previous owner(s) in KM)", min_value=0)
    fuel_type = st.selectbox("Fuel Type (The type of fuel used by the car)", ("Petrol", "Diesel", "Electric", "CNG", "LPG"))
    transmission = st.selectbox("Transmission (The type of transmission used by the car)", ("Automatic", "Manual"))
    owner_type = st.selectbox("Owner Type (Whether the ownership is Firsthand, Second hand or other)", ("First", "Second", "Third", "Fourth & Above"))

    # Accept mileage as text input to handle float values and then convert to float, shows error message if the user inputs alphabets
    mileage_input = st.text_input("Mileage (The standard mileage offered by the car company in kilometers per liter (kmpl) or kilometers per kilogram (km/kg))")

    try:
        mileage = float(mileage_input) if mileage_input else None
    except ValueError:
        st.error("Please enter a valid decimal number for mileage.")

    engine = st.number_input("Engine (The displacement volume of the engine in cubic centimeter (cc))", min_value=0)

    # Accept power as text input to handle float values and then convert to float, shows error message if the user inputs alphabets
    power_input = st.text_input("Power (The maximum power of the engine in brake horsepower (bhp))")

    try:
        power = float(power_input) if power_input else None
    except ValueError:
        st.error("Please enter a valid decimal number for power.")

    seats = st.number_input("Seats (The number of seats in the car) (Please select between 1 and 10 inclusive)", min_value=1, max_value=10, step=1)

    # Custom CSS for Predict button
    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            background-color: blue;
            color: white;
            font-size: 18px;
            border-radius: 10px;
            height: 3em;
            width: 100%;
        }
        div.stButton > button:hover {
            background-color: #4169E1;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    if st.button("🚗 Predict Price"):
        # Validation for location
        if not location.strip():
            st.error("Please enter a location.")
            st.stop()
        
        # Collect input features into DataFrame
        input_data = pd.DataFrame([{
            "Location": location,
            "Year": year,
            "Kilometers_Driven": kilometers_driven,
            "Fuel_Type": fuel_type,
            "Transmission": transmission,
            "Owner_Type": owner_type,
            "Mileage": mileage,
            "Engine": engine,
            "Power": power,
            "Seats": seats
        }])

        # Make prediction using pipeline
        try:
            prediction = model.predict(input_data)
            st.success(f"Predicted Price is {prediction[0]:.2f} Lakh in Indian Rupees (INR).")
        except Exception as e:
            st.error(f"Error while predicting: {e}")

# Batch Prediction
elif prediction_type == "Batch":
    st.subheader("Batch Prediction")
    st.write("Upload a CSV or Excel file containing car details for multiple predictions. The file should contain the following columns:")
    
    # Display expected columns with descriptions matching real-time inputs
    expected_columns = [
        "Location", "Year", "Kilometers_Driven", "Fuel_Type", 
        "Transmission", "Owner_Type", "Mileage", "Engine", "Power", "Seats"
    ]
    
    column_descriptions = [
        "Location (The location in which the car is being sold or is available for purchase)",
        "Year (The year or edition of the model) (1998-2019)",
        "Kilometers_Driven (The total kilometres driven in the car by the previous owner(s) in KM)",
        "Fuel_Type (The type of fuel used by the car)",
        "Transmission (The type of transmission used by the car)",
        "Owner_Type (Whether the ownership is Firsthand, Second hand or other)",
        "Mileage (The standard mileage offered by the car company in kilometers per liter (kmpl) or kilometers per kilogram (km/kg))",
        "Engine (The displacement volume of the engine in cubic centimeter (cc))",
        "Power (The maximum power of the engine in brake horsepower (bhp))",
        "Seats (The number of seats in the car) (1-10 inclusive)"
    ]
    
    st.write("**Required columns:**")
    for i, desc in enumerate(column_descriptions, 1):
        st.write(f"{i}. {desc}")
    
    st.write("**Important notes:**")
    st.write("- Year should be between 1998-2019 inclusive.")
    st.write("- Seats should be between 1-10 inclusive.") 
    st.write("- Fuel_Type should be one of these types: Petrol, Diesel, Electric, CNG, LPG.")
    st.write("- Transmission should be either Manual or Automatic.")
    st.write("- Owner_Type should be one of these types: First, Second, Third, Fourth & Above.")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    # Clear previous results when no file is uploaded or when a new file is uploaded
    if uploaded_file is None:
        # Clear session state when file is removed
        if 'prediction_results' in st.session_state:
            del st.session_state['prediction_results']
        if 'predictions' in st.session_state:
            del st.session_state['predictions']
        if 'current_file_name' in st.session_state:
            del st.session_state['current_file_name']
    else:
        # Clear session state when a different file is uploaded
        if 'current_file_name' in st.session_state and st.session_state['current_file_name'] != uploaded_file.name:
            if 'prediction_results' in st.session_state:
                del st.session_state['prediction_results']
            if 'predictions' in st.session_state:
                del st.session_state['predictions']
        
        # Store current file name
        st.session_state['current_file_name'] = uploaded_file.name
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file based on its extension
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")
                st.stop()
            
            st.subheader("Uploaded Data Preview")
            st.write(f"Total records: {len(df)}")
            st.dataframe(df.head())
            
            # Check if all required columns are present
            missing_columns = [col for col in expected_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
            else:
                # Validate data ranges
                validation_errors = []
                
                # Check year range
                invalid_years = df[(df['Year'] < 1998) | (df['Year'] > 2019)]
                if not invalid_years.empty:
                    validation_errors.append(f"Found {len(invalid_years)} records with invalid years (should be 1998-2019)")
                
                # Check seats range
                invalid_seats = df[(df['Seats'] < 1) | (df['Seats'] > 10)]
                if not invalid_seats.empty:
                    validation_errors.append(f"Found {len(invalid_seats)} records with invalid seats (should be 1-10)")
                
                # Check for missing values in critical columns
                missing_data = df[expected_columns].isnull().sum()
                cols_with_missing = missing_data[missing_data > 0]
                if not cols_with_missing.empty:
                    validation_errors.append(f"Missing data found in columns: {dict(cols_with_missing)}")
                
                if validation_errors:
                    st.warning("Data validation issues found:")
                    for error in validation_errors:
                        st.warning(f"{error}")
                    st.write("Please fix the data issues before proceeding with predictions.")
                else:
                    st.success("Data is successfully validated. You can now proceed to make predictions.")
                
                # Custom CSS for Predict button (same as single prediction)
                st.markdown(
                    """
                    <style>
                    div.stButton > button:first-child {
                        background-color: blue;
                        color: white;
                        font-size: 18px;
                        border-radius: 10px;
                        height: 3em;
                        width: 100%;
                    }
                    div.stButton > button:hover {
                        background-color: #4169E1;
                        color: white;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                 
                # Predict button for batch prediction
                if st.button("🚗 Predict All Prices"):
                    if validation_errors:
                        st.error("Cannot proceed with predictions due to data validation errors.")
                    else:
                        try:
                            # Make predictions for all records
                            with st.spinner("Making predictions..."):
                                predictions = model.predict(df[expected_columns])
                            
                            # Add predictions to the dataframe
                            df_results = df.copy()
                            df_results['Predicted_Price_Lakh'] = predictions.round(2)
                            
                            # Store results in session state to persist across reruns
                            st.session_state['prediction_results'] = df_results
                            st.session_state['predictions'] = predictions
                            
                        except Exception as e:
                            st.error(f"Error during prediction: {e}")
                
                # Display results if they exist in session state
                if 'prediction_results' in st.session_state and 'predictions' in st.session_state:
                    df_results = st.session_state['prediction_results']
                    predictions = st.session_state['predictions']
                    
                    st.subheader("Prediction Results")
                    st.write(f"Successfully predicted prices for {len(df_results)} used cars.")
                    
                    # Display results
                    st.dataframe(df_results)
                    
                    # Summary statistics - made smaller
                    st.markdown("#### Summary Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(label="Average Price (INR/Lakh)", value=f"{predictions.mean():.2f}")

                    with col2:
                        st.metric(label="Minimum Price (INR/Lakh)", value=f"{predictions.min():.2f}")

                    with col3:
                        st.metric(label="Maximum Price (INR/Lakh)", value=f"{predictions.max():.2f}")

                    
                    # Download results
                    csv = df_results.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="Used_Car_Price_Predictions.csv",
                        mime="text/csv",
                        key="download_csv"
                    )
        
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            st.write("Please make sure your file is in valid CSV or Excel file format.")
    
    else:
        st.info("Please upload a CSV or Excel file to get started with batch predictions.")