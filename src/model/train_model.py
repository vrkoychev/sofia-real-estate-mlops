import joblib
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Paths ---
PROCESSED_DATA_PATH = "sofia-real-estate-mlops/datasets/processed/processed_data.pkl"
PREPROCESSOR_PATH = "sofia-real-estate-mlops/datasets/processed/preprocessor.pkl"
MODEL_PATH = "sofia-real-estate-mlops/models/best_model.pkl"
FULL_PIPELINE_PATH = "sofia-real-estate-mlops/models/full_pipeline.pkl"

# --- Load processed features ---
X_train, X_test, y_train, y_test = joblib.load(PROCESSED_DATA_PATH)

# --- Define models ---
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
    "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42),
    "XGBoost": XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        objective='reg:squarederror'
    )
}

# --- Function to train, evaluate, and return best model ---
def train_and_select_best_model(models, X_train, y_train, X_test, y_test):
    best_model = None
    best_mae = float('inf')
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        print(f"--- {name} ---")
        print(f"MAE: {mae:,.0f}")
        print(f"RMSE: {rmse:,.0f}")
        print(f"R²: {r2:.3f}\n")
        
        if mae < best_mae:
            best_mae = mae
            best_model = model
    
    return best_model

# --- Train and select best model ---
best_model = train_and_select_best_model(models, X_train, y_train, X_test, y_test)

# --- Save best model ---
joblib.dump(best_model, MODEL_PATH)
print(f"✅ Best model saved at {MODEL_PATH}")

from sklearn.pipeline import Pipeline

preprocessor = joblib.load(PREPROCESSOR_PATH)

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', best_model)
])

# Save the full pipeline
joblib.dump(full_pipeline, FULL_PIPELINE_PATH)
print(f"✅ Full pipeline saved at {FULL_PIPELINE_PATH}")

