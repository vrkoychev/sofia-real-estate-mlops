import subprocess
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

    # Get the absolute path to the scrapy project directory
    current_dir = Path(__file__).parent.absolute()
    project_root = current_dir.parent
    scrapy_project_dir = current_dir.parent / "scraping"
    
    # Verify the directory exists
    if not scrapy_project_dir.exists() or not scrapy_project_dir.is_dir():
        logger.error(f"Scrapy project directory not found: {scrapy_project_dir}")
        raise NotADirectoryError(f"The path {scrapy_project_dir} is not a valid directory")

    # Construct the command to run the spider
    command = [
        "scrapy", "crawl", "alo",
        "-O", output_path,
        "--nolog"
    ]

    logger.info(f"Starting spider. Output will be saved to: {output_path}")
    logger.info(f"Running command from directory: {scrapy_project_dir}")
    
    try:
        result = subprocess.run(
            command,
            cwd=str(scrapy_project_dir),
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
    
    # Convert to absolute path relative to project root
    current_dir = Path(__file__).parent.absolute()
    project_root = current_dir.parent
    output_path = project_root / args.output
    
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    run_spider(str(output_path))

if __name__ == "__main__":
    main()