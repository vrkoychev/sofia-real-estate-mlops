from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(input_path: str) -> pd.DataFrame:
    """Load cleaned dataset from CSV file."""
    logger.info(f"Loading cleaned data from {input_path}")
    return pd.read_csv(input_path)

def handle_missing_values(df: pd.DataFrame, numeric_cols: list, categorical_cols: list) -> pd.DataFrame:
    """Handle missing values in the dataset."""
    logger.info("Handling missing values")
    
    # Fill numeric NaNs with median
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Fill categorical NaNs with 'Unknown'
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')
    
    return df

def handle_outliers(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:
    """Handle outliers using IQR method (cap at bounds)."""
    logger.info("Handling outliers using IQR method")
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        # Cap values at lower and upper bounds
        df[col] = df[col].clip(lower=lower, upper=upper)
    
    return df

def create_preprocessor(categorical_cols: list, numeric_cols: list) -> ColumnTransformer:
    """Create and return a column transformer for preprocessing."""
    logger.info("Creating preprocessor")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numeric_cols)
        ]
    )
    
    return preprocessor

def split_data(df_features: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
    """Split data into training and testing sets."""
    logger.info("Splitting data into training and testing sets")
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df_features, y, test_size=test_size, random_state=random_state
    )
    
    return X_train_raw, X_test_raw, y_train, y_test

def save_artifacts(preprocessor, X_train, X_test, y_train, y_test, preprocessor_path, processed_data_path):
    """Save preprocessor and processed data."""
    logger.info("Saving preprocessor and processed data")
    
    # Ensure output directories exist
    Path(preprocessor_path).parent.mkdir(parents=True, exist_ok=True)
    Path(processed_data_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save artifacts
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump((X_train, X_test, y_train, y_test), processed_data_path)

def prepare_features(input_path: str, preprocessor_path: str, processed_data_path: str):
    """
    Main function to prepare features for modeling.
    
    Args:
        input_path: Path to the cleaned CSV file
        preprocessor_path: Path where the preprocessor should be saved
        processed_data_path: Path where the processed data should be saved
    """
    try:
        # Load data
        df = load_data(input_path)
        
        # Define column types
        numeric_cols = ['size', 'floor_number', 'year_built']
        categorical_cols = ['location', 'property_type', 'construction_type', 'floor_type']
        
        # Drop rows where target 'price' is NaN
        df = df.dropna(subset=['price'])
        
        # Drop columns not needed as raw features
        df_features = df.drop(columns=['price'])
        
        # Handle missing values
        df_features = handle_missing_values(df_features, numeric_cols, categorical_cols)
        
        # Handle outliers (including price)
        df_with_price = handle_outliers(df, numeric_cols + ['price'])
        y = df_with_price['price']
        
        # Split data
        X_train_raw, X_test_raw, y_train, y_test = split_data(df_features, y)
        
        # Create and fit preprocessor
        preprocessor = create_preprocessor(categorical_cols, numeric_cols)
        X_train = preprocessor.fit_transform(X_train_raw)
        X_test = preprocessor.transform(X_test_raw)
        
        # Save artifacts
        save_artifacts(preprocessor, X_train, X_test, y_train, y_test, 
                      preprocessor_path, processed_data_path)
        
        logger.info("✅ Features prepared, missing values handled, outliers capped, and saved for modeling.")
        
    except Exception as e:
        logger.error(f"Feature preparation failed: {e}")
        raise

def main():
    """Main function to run the feature preparation pipeline from command line."""
    parser = argparse.ArgumentParser(description='Prepare features for modeling')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to input cleaned data CSV file')
    parser.add_argument('--preprocessor', type=str, required=True,
                        help='Path to output preprocessor PKL file')
    parser.add_argument('--processed-data', type=str, required=True,
                        help='Path to output processed data PKL file')
    
    args = parser.parse_args()
    
    # Run the feature preparation pipeline
    prepare_features(
        input_path=args.input,
        preprocessor_path=args.preprocessor,
        processed_data_path=args.processed_data
    )

if __name__ == "__main__":
    main()
