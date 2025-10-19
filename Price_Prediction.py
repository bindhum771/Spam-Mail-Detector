import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# --- UPDATED IMPORTS ---
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# --- Configuration and Persistence ---
MODEL_PATH = './house_price_model_rf.pkl' # Changed name for new model type
SCALER_PATH = './house_price_scaler_rf.pkl' # This will now store the full ColumnTransformer pipeline
METRICS_PATH = './house_price_metrics_rf.pkl'
DF_PATH = './processed_data.pkl'

# --- Utility Function to Clear Artifacts ---

def clear_artifacts():
    """Deletes existing model, scaler, and metrics files."""
    for path in [MODEL_PATH, SCALER_PATH, METRICS_PATH, DF_PATH]:
        if os.path.exists(path):
            os.remove(path)
            # st.warning(f"Cleared old artifact: {path}") # Commented out to reduce clutter


# --- 1. Data Loading and Preparation (Using train.csv) ---

@st.cache_data
def load_and_preprocess_data():
    """
    Loads data from train.csv, handles missing values, scales the target price,
    and defines the preprocessing pipeline using ColumnTransformer.
    """
    st.info("📊 Loading and Preprocessing data from train.csv...")
    
    # 1. Load the dataset
    try:
        # ------------------------------------------------------------------
        # FIX: Check if the file exists before attempting to read
        # ------------------------------------------------------------------
        if not os.path.exists('train.csv'):
            st.error("FATAL ERROR: 'train.csv' was not found in the current working directory.")
            st.warning("Please ensure the file is uploaded and available to the running script.")
            return None, None, None, None, None, None
            
        df = pd.read_csv('train.csv')
    except PermissionError as e:
        # ------------------------------------------------------------------
        # FIX: Handle explicit PermissionError
        # ------------------------------------------------------------------
        st.error(f"Permission denied when reading 'train.csv'. Error: {e}")
        st.warning("This is likely an environment issue. Please try refreshing or re-running the application to reset file permissions.")
        return None, None, None, None, None, None
    except FileNotFoundError:
        # This block should now rarely hit due to the check above
        st.error("Error reading 'train.csv'. Please check the file name and path.")
        return None, None, None, None, None, None
        
    # 2. Define Features (X) and Target (y)
    
    # --- IMPROVED FEATURE SELECTION ---
    # Using more features and the categorical 'POSTED_BY'
    NUMERIC_FEATURES = ['SQUARE_FT', 'BHK_NO.', 'LONGITUDE', 'LATITUDE']
    CATEGORICAL_FEATURES = ['POSTED_BY']
    BINARY_FEATURES = ['READY_TO_MOVE', 'RESALE']
    
    ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES
    target_name = 'TARGET(PRICE_IN_LACS)'

    # Filter columns and handle NaNs
    df = df[ALL_FEATURES + [target_name]].copy()
    df.dropna(inplace=True)
    
    # 3. Scale Target: Convert price from Lakhs to Rupees (₹)
    df['Price'] = df[target_name] * 100000
    
    X = df[ALL_FEATURES]
    y = df['Price']
    
    st.success(f"Dataset loaded. Total samples after cleaning: {len(X)}")

    # 4. Create Preprocessing Pipeline (ColumnTransformer)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
            ('bin', 'passthrough', BINARY_FEATURES)
        ],
        remainder='drop'
    )

    # Data Splitting
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Fit and transform the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save the feature names (including the new one-hot encoded ones)
    new_feature_names = preprocessor.get_feature_names_out()
    
    # We save the raw data for plotting later against the target
    joblib.dump(df, DF_PATH) 

    return X_train_processed, X_test_processed, y_train, y_test, preprocessor, new_feature_names


# --- 2. Model Training and Saving ---

@st.cache_resource
def train_and_save_model(_X_train, _X_test, _y_train, _y_test, _preprocessor, feature_names):
    """
    Trains the Random Forest Regressor model, evaluates it, and saves model artifacts.
    """
    st.info("🧠 Training Random Forest Regressor Model...")
    
    # --- MODEL UPGRADE: Random Forest Regressor ---
    model = RandomForestRegressor(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1, # Use all available cores
        max_depth=10 # Simple depth for faster training
    )
    model.fit(_X_train, _y_train)

    # Evaluation
    y_pred = model.predict(_X_test)
    
    # Calculate Metrics
    mse = mean_squared_error(_y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(_y_test, y_pred)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'y_test': _y_test.tolist(),
        'y_pred': y_pred.tolist()
    }

    # Save Model Artifacts
    try:
        joblib.dump(model, MODEL_PATH)
        joblib.dump(_preprocessor, SCALER_PATH) # Saving the preprocessor pipeline
        joblib.dump(metrics, METRICS_PATH)
        st.success("Model, preprocessor pipeline, and metrics saved!")
    except Exception as e:
        st.error(f"Error saving model artifacts: {e}")

    return model, _preprocessor, metrics

# --- 3. Model Loading ---

def load_model_artifacts():
    """Loads saved model, preprocessor, and metrics, or initiates training if they don't exist."""
    
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(METRICS_PATH):
        st.info("Loading trained model and artifacts from disk...", icon="💾")
        try:
            model = joblib.load(MODEL_PATH)
            preprocessor = joblib.load(SCALER_PATH)
            metrics = joblib.load(METRICS_PATH)
            return model, preprocessor, metrics
        except Exception as e:
            st.error(f"Corrupted saved files found ({e}). Forcing retraining.")
            clear_artifacts()
            
    # If loading failed or files don't exist, train the model
    X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names_out = load_and_preprocess_data()
    
    # ------------------------------------------------------------------
    # FIX: Check if data loading was successful before training
    # ------------------------------------------------------------------
    if X_train_processed is None:
        # If any preprocessing step returned None (due to file or permission error), stop here.
        st.stop()
        
    return train_and_save_model(X_train_processed, X_test_processed, y_train, y_test, preprocessor, feature_names_out)


# --- 4. Prediction Function ---

def predict_price(model, preprocessor, input_dict):
    """
    Takes user input dictionary, converts it to DataFrame, processes it
    using the ColumnTransformer, and returns a price prediction.
    """
    # 1. Convert input dictionary to a DataFrame for processing
    input_df = pd.DataFrame([input_dict])
    
    # 2. Process the input using the fitted ColumnTransformer
    input_processed = preprocessor.transform(input_df)
    
    # 3. Predict the price
    prediction = model.predict(input_processed)[0]
    
    # Ensure prediction is non-negative
    return max(0, prediction)


# --- 5. Visualization Functions (No change) ---

def plot_prediction_vs_actual(metrics):
    """Plots true vs. predicted prices for the test set."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    y_test = np.array(metrics['y_test'])
    y_pred = np.array(metrics['y_pred'])
    
    sns.scatterplot(x=y_test, y=y_pred, ax=ax, alpha=0.6)
    
    # Add perfect prediction line (y=x)
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

    ax.set_title('True Price vs. Predicted Price (Test Set)', fontsize=14)
    ax.set_xlabel('True Price (₹)', fontsize=12)
    ax.set_ylabel('Predicted Price (₹)', fontsize=12)
    plt.close(fig)
    return fig

# --- Main Streamlit Application ---

# Load or Train Model (This loads Random Forest and the ColumnTransformer)
model, preprocessor, metrics = load_model_artifacts()

st.title("🏡 AI House Price Predictor (High Accuracy Model)")
st.markdown("### Using Random Forest Regressor and Comprehensive Features from `train.csv`")

# --- Model Metrics Display ---
st.subheader("Model Performance on Test Data")
col1, col2, col3 = st.columns(3)

col1.metric("Mean Squared Error (MSE)", f"{metrics['MSE']:.2f}")
col2.metric("Root Mean Squared Error (RMSE)", f"{metrics['RMSE']:.2f}")
col3.metric("R-Squared ($R^2$)", f"{metrics['R2']:.4f}")

st.caption("Random Forest generally achieves a higher $R^2$ score than Linear Regression.")

# --- Real-Time Prediction UI ---
st.markdown("---")
st.subheader("Real-Time House Price Prediction")
st.markdown("Adjust the house features below to get an instant price forecast.")

# Define feature ranges for the UI based on general market sense
SQFT_MIN, SQFT_MAX, SQFT_DEFAULT = 300, 6000, 1200
BHK_MIN, BHK_MAX, BHK_DEFAULT = 1, 6, 2
LAT_MIN, LAT_MAX, LAT_DEFAULT = 8.0, 36.0, 20.0 # Approximate range for India
LON_MIN, LON_MAX, LON_DEFAULT = 68.0, 97.0, 80.0 # Approximate range for India

with st.form("prediction_form"):
    
    st.markdown("#### House Features")
    
    # Feature 1: SQUARE_FT 
    sq_footage = st.slider("Square Footage (SqFt)", min_value=SQFT_MIN, max_value=SQFT_MAX, value=SQFT_DEFAULT, step=50)
    
    col_b_1, col_b_2 = st.columns(2)
    
    # Feature 2: BHK_NO. 
    bedrooms = col_b_1.slider("BHK No. (Bedrooms)", min_value=BHK_MIN, max_value=BHK_MAX, value=BHK_DEFAULT, step=1)
    
    # Feature 3: POSTED_BY (Categorical input)
    posted_by = col_b_2.selectbox("Posted By", options=['Owner', 'Dealer', 'Builder'], index=0)

    st.markdown("#### Location and Status")
    
    col_l_1, col_l_2 = st.columns(2)
    
    # Feature 4 & 5: LONGITUDE and LATITUDE
    longitude = col_l_1.slider("Longitude", min_value=LON_MIN, max_value=LON_MAX, value=LON_DEFAULT, step=0.1, format="%0.2f")
    latitude = col_l_2.slider("Latitude", min_value=LAT_MIN, max_value=LAT_MAX, value=LAT_DEFAULT, step=0.1, format="%0.2f")

    col_s_1, col_s_2 = st.columns(2)
    
    # Feature 6: READY_TO_MOVE (Binary)
    ready_to_move_flag = col_s_1.selectbox("Ready to Move?", options=[1, 0], format_func=lambda x: "Yes (1)" if x == 1 else "No (0 - Under Construction)")

    # Feature 7: RESALE (Binary)
    resale_flag = col_s_2.selectbox("Is this a Resale Property?", options=[1, 0], format_func=lambda x: "Yes (1)" if x == 1 else "No (0 - New)")

    # Form submission button
    submitted = st.form_submit_button("Predict Price", type="primary")

    if submitted:
        # Prepare feature dictionary, MUST match the ColumnTransformer's feature order
        input_data_dict = {
            'SQUARE_FT': sq_footage,
            'BHK_NO.': bedrooms,
            'LONGITUDE': longitude,
            'LATITUDE': latitude,
            'POSTED_BY': posted_by,
            'READY_TO_MOVE': ready_to_move_flag,
            'RESALE': resale_flag
        }
        
        # Get prediction
        predicted_value = predict_price(model, preprocessor, input_data_dict)
        
        # Display the result
        st.markdown(
            f"""
            <div style='background-color: #e0f2fe; padding: 20px; border-radius: 10px; text-align: center; border: 2px solid #0284c7;'>
                <h3 style='color: #0284c7; margin-bottom: 5px;'>Predicted House Price</h3>
                <p style='font-size: 3em; font-weight: bold; color: #075985;'>₹{predicted_value:,.2f}</p>
            </div>
            """, 
            unsafe_allow_html=True
        )


# --- Model Visualization ---
st.markdown("---")
st.subheader("Model Validation")
st.pyplot(plot_prediction_vs_actual(metrics))

st.caption("Points clustered near the red dashed line indicate accurate predictions.")
