from pathlib import Path
import pandas as pd
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(input_path: str) -> pd.DataFrame:
    """Load raw dataset from CSV file."""
    logger.info(f"Loading raw data from {input_path}")
    return pd.read_csv(input_path)

def clean_location(df: pd.DataFrame) -> pd.DataFrame:
    """Clean location column: keep only text before comma."""
    logger.info("Cleaning location column")
    df['location'] = df['location'].str.split(',').str[0].str.strip()
    return df

def clean_price(df: pd.DataFrame) -> pd.DataFrame:
    """Clean price column: keep only numbers, remove spaces and €."""
    logger.info("Cleaning price column")
    df['price'] = df['price'].str.replace(r'[^0-9]', '', regex=True)
    df['price'] = pd.to_numeric(df['price'], errors='coerce').astype('Int64')
    return df

def clean_property_type(df: pd.DataFrame) -> pd.DataFrame:
    """Clean property type column: keep only first word."""
    logger.info("Cleaning property type column")
    df['property_type'] = df['property_type'].str.split().str[0]
    return df

def clean_size(df: pd.DataFrame) -> pd.DataFrame:
    """Clean size column: keep only numbers."""
    logger.info("Cleaning size column")
    df['size'] = df['size'].str.extract(r'(\d+)')
    df['size'] = pd.to_numeric(df['size'], errors='coerce').astype('Int64')
    return df

def clean_year_built(df: pd.DataFrame) -> pd.DataFrame:
    """Clean year built column: keep only numbers."""
    logger.info("Cleaning year built column")
    df['year_built'] = df['year_built'].str.extract(r'(\d{4})')
    df['year_built'] = pd.to_numeric(df['year_built'], errors='coerce').astype('Int64')
    return df

def clean_floor_number(df: pd.DataFrame) -> pd.DataFrame:
    """Clean floor number column: keep only numbers, map Партер to 0."""
    logger.info("Cleaning floor number column")
    df['floor_number'] = df['floor_number'].replace("Партер", "0")
    df['floor_number'] = df['floor_number'].str.extract(r'(-?\d+)')
    df['floor_number'] = pd.to_numeric(df['floor_number'], errors='coerce').astype('Int64')
    return df

def save_data(df: pd.DataFrame, output_path: str):
    """Save cleaned dataset to CSV file."""
    logger.info(f"Saving cleaned data to {output_path}")
    df.to_csv(output_path, index=False)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all cleaning operations to the dataframe.
    
    Args:
        df: Raw dataframe to clean
        
    Returns:
        Cleaned dataframe
    """
    logger.info("Starting data cleaning process")
    
    # Make a copy to avoid modifying the original dataframe
    cleaned_df = df.copy()
    
    # Apply all cleaning functions
    cleaned_df = clean_location(cleaned_df)
    cleaned_df = clean_price(cleaned_df)
    cleaned_df = clean_property_type(cleaned_df)
    cleaned_df = clean_size(cleaned_df)
    cleaned_df = clean_year_built(cleaned_df)
    cleaned_df = clean_floor_number(cleaned_df)
    
    logger.info("Data cleaning completed")
    return cleaned_df

def main(input_path: str, output_path: str):
    """
    Main function to run the entire preprocessing pipeline.
    
    Args:
        input_path: Path to the raw CSV file
        output_path: Path where the cleaned CSV should be saved
    """
    try:
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Load, clean, and save data
        raw_df = load_data(input_path)
        cleaned_df = clean_data(raw_df)
        save_data(cleaned_df, output_path)
        
        logger.info(f"Preprocessing complete. Saved to {output_path}")
        logger.info(f"Data types:\n{cleaned_df.dtypes}")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Preprocess real estate data')
    parser.add_argument('--input', type=str, required=True, 
                        help='Path to input raw data CSV file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to output cleaned CSV file')
    
    args = parser.parse_args()
    
    # Run the main function with provided arguments
    main(input_path=args.input, output_path=args.output)