from pathlib import Path
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import argparse
import logging

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_mlflow_experiment(experiment_name: str = "Sofia-Real-Estate"):
    """Set up MLflow experiment."""
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)

def load_processed_data(processed_data_path: str):
    """Load processed data from pickle file."""
    logger.info(f"Loading processed data from {processed_data_path}")
    return joblib.load(processed_data_path)

def load_preprocessor(preprocessor_path: str):
    """Load preprocessor from pickle file."""
    logger.info(f"Loading preprocessor from {preprocessor_path}")
    return joblib.load(preprocessor_path)

def define_models():
    """Define the models to train and evaluate."""
    logger.info("Defining models to train")
    
    models = {
        "LinearRegression": LinearRegression(),
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
    
    return models

def log_model_metrics(mlflow_client, run_id, model_name, metrics):
    """Log model metrics to MLflow."""
    mlflow_client.log_metric(run_id, f"{model_name}_MAE", metrics["mae"])
    mlflow_client.log_metric(run_id, f"{model_name}_RMSE", metrics["rmse"])
    mlflow_client.log_metric(run_id, f"{model_name}_R2", metrics["r2"])

def evaluate_model(model, X_test, y_test, model_name: str):
    """Evaluate a model and return metrics."""
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"--- {model_name} ---")
    logger.info(f"MAE: {mae:,.0f}")
    logger.info(f"RMSE: {rmse:,.0f}")
    logger.info(f"R²: {r2:.3f}")
    
    return {
        "model_name": model_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }

def train_and_evaluate_models(models, X_train, y_train, X_test, y_test):
    """Train and evaluate all models, returning the best one."""
    logger.info("Training and evaluating models")
    
    best_model = None
    best_metrics = None
    best_model_name = None
    all_metrics = []
    
    # Start MLflow run
    with mlflow.start_run(run_name="Model_Training_Comparison"):
        mlflow.log_param("total_models", len(models))
        
        for name, model in models.items():
            logger.info(f"Training {name}")
            
            # Start a nested run for each model
            with mlflow.start_run(nested=True, run_name=f"Model_{name}"):
                # Log model parameters
                if hasattr(model, "get_params"):
                    mlflow.log_params(model.get_params())
                
                # Train the model
                model.fit(X_train, y_train)
                
                # Evaluate the model
                metrics = evaluate_model(model, X_test, y_test, name)
                all_metrics.append(metrics)
                
                # Log metrics
                mlflow.log_metric("MAE", metrics["mae"])
                mlflow.log_metric("RMSE", metrics["rmse"])
                mlflow.log_metric("R2", metrics["r2"])
                
                # Create signature and input example for each model
                model_signature = infer_signature(X_train, model.predict(X_train))
                model_input_example = X_train[:2]

                # Log model with signature and input example
                mlflow.sklearn.log_model(
                    sk_model=model,
                    name=f"model_{name}",
                    signature=model_signature,
                    input_example=model_input_example
                )
                
                # Check if this is the best model so far
                if best_model is None or metrics["mae"] < best_metrics["mae"]:
                    best_model = model
                    best_metrics = metrics
                    best_model_name = name
        
        # Log the best model
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("best_MAE", best_metrics["mae"])
        mlflow.log_metric("best_RMSE", best_metrics["rmse"])
        mlflow.log_metric("best_R2", best_metrics["r2"])
    
    logger.info(f"Best model: {best_model_name} with MAE: {best_metrics['mae']:,.0f}")

    return best_model, best_model_name, all_metrics

def save_model(model, model_path: str):
    """Save a model to a file."""
    logger.info(f"Saving model to {model_path}")
    
    # Ensure the output directory exists
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, model_path)

def create_full_pipeline(preprocessor, model):
    """Create a full pipeline combining preprocessor and model."""
    logger.info("Creating full pipeline")
    
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    return full_pipeline

def register_best_model(best_model, best_model_name, X_train):
    """Register the best model in MLflow Model Registry."""
    logger.info(f"Registering best model: {best_model_name}")
    
    # Log the best model in a separate run
    with mlflow.start_run(run_name=f"Best_Model_{best_model_name}"):
        # Log parameters and metrics
        if hasattr(best_model, "get_params"):
            mlflow.log_params(best_model.get_params())
        
        # Create a signature for the model
        signature = infer_signature(X_train, best_model.predict(X_train))
        
        # Create an input example (first 5 rows of training data)
        input_example = X_train[:5]
        
        # Log the model with all details
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            signature=signature,
            input_example=input_example,
            registered_model_name="Sofia-Real-Estate-Predictor"
        )
        
        # Add model description
        mlflow.set_tag("model_description", f"Best model for Sofia Real Estate Price Prediction: {best_model_name}")

def train_model(processed_data_path: str, preprocessor_path: str, 
                model_path: str, full_pipeline_path: str):
    """
    Main function to train and select the best model.
    
    Args:
        processed_data_path: Path to the processed data pickle file
        preprocessor_path: Path to the preprocessor pickle file
        model_path: Path where the best model should be saved
        full_pipeline_path: Path where the full pipeline should be saved
    """
    try:
        # Setup MLflow
        setup_mlflow_experiment()

        # Load data and preprocessor
        X_train, X_test, y_train, y_test = load_processed_data(processed_data_path)
        preprocessor = load_preprocessor(preprocessor_path)
        
        # Define and train models
        models = define_models()
        best_model, best_model_name, all_metrics = train_and_evaluate_models(
            models, X_train, y_train, X_test, y_test
        )
        
        # Save the best model
        save_model(best_model, model_path)
        
        # Create and save the full pipeline
        full_pipeline = create_full_pipeline(preprocessor, best_model)
        save_model(full_pipeline, full_pipeline_path)

        # Register the best model in MLflow
        register_best_model(best_model, best_model_name, X_train)
        
        logger.info("✅ Model training completed successfully with MLflow tracking")
        
        # Print summary of all models' performance
        logger.info("\n=== Model Performance Summary ===")
        for metrics in all_metrics:
            logger.info(f"{metrics['model_name']}: MAE={metrics['mae']:,.0f}, R²={metrics['r2']:.3f}")
        
        return best_model_name, all_metrics
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise

def main():
    """Main function to run the model training pipeline from command line."""
    parser = argparse.ArgumentParser(description='Train and select the best model')
    parser.add_argument('--processed-data', type=str, required=True,
                        help='Path to processed data PKL file')
    parser.add_argument('--preprocessor', type=str, required=True,
                        help='Path to preprocessor PKL file')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to output model PKL file')
    parser.add_argument('--full-pipeline', type=str, required=True,
                        help='Path to output full pipeline PKL file')
    
    args = parser.parse_args()
    
    # Run the model training pipeline
    train_model(
        processed_data_path=args.processed_data,
        preprocessor_path=args.preprocessor,
        model_path=args.model,
        full_pipeline_path=args.full_pipeline
    )

if __name__ == "__main__":
    main()