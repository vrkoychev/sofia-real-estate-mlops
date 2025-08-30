from pathlib import Path
import argparse
import logging
import sys
from prefect import flow, task

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.run_spider import run_spider
from src.preprocess_raw import main as preprocess_data
from src.prepare_features import prepare_features
from src.train_model import train_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@task(name="Scrape Data", retries=2, retry_delay_seconds=60)
def scrape_task(raw_data_path: str):
    """Task to run the Scrapy spider and collect new data."""
    logger.info(f"Starting data scraping task. Output: {raw_data_path}")
    run_spider(raw_data_path)
    return raw_data_path

@task(name="Preprocess Data")
def preprocess_task(raw_data_path: str, clean_data_path: str):
    """Task to clean the raw data."""
    logger.info(f"Starting data preprocessing task. Input: {raw_data_path}, Output: {clean_data_path}")
    preprocess_data(input_path=raw_data_path, output_path=clean_data_path)
    return clean_data_path

@task(name="Prepare Features")
def prepare_features_task(clean_data_path: str, preprocessor_path: str, processed_data_path: str):
    """Task to prepare features for modeling."""
    logger.info(f"Starting feature preparation task. Input: {clean_data_path}")
    prepare_features(
        input_path=clean_data_path,
        preprocessor_path=preprocessor_path,
        processed_data_path=processed_data_path
    )
    return processed_data_path

@task(name="Train Model")
def train_model_task(processed_data_path: str, preprocessor_path: str, 
                    model_path: str, full_pipeline_path: str):
    """Task to train and select the best model."""
    logger.info(f"Starting model training task. Input: {processed_data_path}")
    train_model(
        processed_data_path=processed_data_path,
        preprocessor_path=preprocessor_path,
        model_path=model_path,
        full_pipeline_path=full_pipeline_path
    )
    return model_path

@flow(name="Sofia-Real-Estate-Pipeline")
def main_flow():
    """The main orchestration flow for the Sofia Real Estate Pipeline."""
    # Define paths
    raw_data_path = "datasets/raw_data.csv"
    clean_data_path = "datasets/clean_data.csv"
    preprocessor_path = "datasets/processed/preprocessor.pkl"
    processed_data_path = "datasets/processed/processed_data.pkl"
    model_path = "models/best_model.pkl"
    full_pipeline_path = "models/full_pipeline.pkl"
    
    # Ensure directories exist
    Path("datasets").mkdir(parents=True, exist_ok=True)
    Path("datasets/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    
    # Execute the pipeline
    scrape_task(raw_data_path)
    preprocess_task(raw_data_path, clean_data_path)
    prepare_features_task(clean_data_path, preprocessor_path, processed_data_path)
    train_model_task(processed_data_path, preprocessor_path, model_path, full_pipeline_path)
    
    logger.info("Pipeline execution completed successfully!")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the Sofia Real Estate Pipeline')
    parser.add_argument('--skip-scraping', action='store_true',
                        help='Skip the scraping step (use existing data)')
    
    args = parser.parse_args()
    
    # Run the flow
    main_flow()