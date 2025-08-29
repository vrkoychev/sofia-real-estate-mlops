import subprocess
import sys
import os
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_spider(output_path: str):
    """
    Runs the Scrapy spider and saves the output to the specified path.

    Args:
        output_path (str): The file path where the scraped data should be saved (e.g., '../data/raw/properties.csv').
    """
    # Construct the command to run the spider
    # The -O flag overwrites the file, --nolog reduces unnecessary console output
    command = [
        "scrapy", "crawl", "alo",
        "-O", output_path,
        "--nolog"
    ]

    logger.info(f"Starting spider. Output will be saved to: {output_path}")
    
    # Change to the directory where the scrapy.cfg file is located
    # Assuming your project structure is: sofia_real_estate_mlops/alo_scraper/
    scrapy_project_dir = os.path.join(os.path.dirname(__file__), '..', 'alo_scraper')
    
    try:
        result = subprocess.run(
            command,
            cwd=scrapy_project_dir,
            capture_output=True,
            text=True,
            check=True
        )
        logger.info("Spider finished successfully!")
        logger.debug(f"Scrapy output: {result.stdout}")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Spider run failed with error code {e.returncode}!")
        logger.error(f"Error output: {e.stderr}")
        raise e

def main():
    """Main function to run the spider from the command line."""
    parser = argparse.ArgumentParser(description='Run the Alo.bg real estate spider')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to the output file (e.g., ../dataset/raw_data.csv)')
    
    args = parser.parse_args()
    
    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    run_spider(args.output)

if __name__ == "__main__":
    main()