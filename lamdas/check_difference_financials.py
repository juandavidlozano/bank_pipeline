import boto3
import logging
from datetime import datetime
import math

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# S3 bucket details
BUCKET_NAME = "alpharank"
FINANCIALS_PREFIX = "financials/"

# Initialize S3 client
s3_client = boto3.client("s3")


def get_latest_financials_folder(bucket_name, prefix):
    """Get the latest date-time folder in the financials S3 bucket."""
    logger.info(f"Fetching the latest folder in S3 bucket: {bucket_name}/{prefix}")
    paginator = s3_client.get_paginator("list_objects_v2")
    response_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter="/")

    latest_folder = None
    latest_file_date = None

    for page in response_iterator:
        if "CommonPrefixes" in page:
            folders = [prefix["Prefix"] for prefix in page["CommonPrefixes"]]
            latest_folder = max(
                folders, key=lambda f: datetime.strptime(f.rstrip("/").split("/")[-1], "%Y-%m-%d_%H-%M-%S")
            )

    if latest_folder:
        logger.info(f"Latest folder found: {latest_folder}")
        # Fetch the latest file date in the folder
        files_response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=latest_folder)
        if "Contents" in files_response:
            latest_file = max(files_response["Contents"], key=lambda f: f["LastModified"])
            latest_file_date = latest_file["LastModified"]
            logger.info(f"Latest file modified date: {latest_file_date}")
    else:
        logger.info("No folders found.")
    return latest_folder, latest_file_date


def is_new_quarter_run_needed(latest_file_date):
    """Check if the current date is outside the latest file's quarter."""
    current_date = datetime.now()
    logger.info(f"Current date: {current_date}")

    # Calculate the quarter for the latest file's date
    latest_file_quarter = math.ceil(latest_file_date.month / 3)
    latest_file_year = latest_file_date.year

    # Calculate the quarter for the current date
    current_quarter = math.ceil(current_date.month / 3)
    current_year = current_date.year

    logger.info(f"Latest file quarter: Q{latest_file_quarter} {latest_file_year}")
    logger.info(f"Current quarter: Q{current_quarter} {current_year}")

    # If the year or quarter has moved forward, return True
    if current_year > latest_file_year or (current_year == latest_file_year and current_quarter > latest_file_quarter):
        return True  # New quarter run is needed
    else:
        return False  # Still in the same quarter


def lambda_handler(event, context):
    """AWS Lambda handler."""
    try:
        # Step 1: Get the latest financials folder and modified date
        latest_folder, latest_file_date = get_latest_financials_folder(BUCKET_NAME, FINANCIALS_PREFIX)
        if not latest_folder or not latest_file_date:
            logger.error("No latest folder or file date found in S3.")
            return {
                "run_needed": True,
                "reason": "No latest folder or file date found."
            }

        # Step 2: Check if a new quarter run is needed
        run_needed = is_new_quarter_run_needed(latest_file_date)

        if run_needed:
            logger.info("New quarter run is needed.")
        else:
            logger.info("No new quarter run is needed.")

        return {
            "run_needed": run_needed,
            "latest_folder": latest_folder,
            "latest_file_date": latest_file_date.strftime("%Y-%m-%dT%H:%M:%S")
        }

    except Exception as e:
        logger.error(f"Error in process: {e}")
        return {
            "error": str(e)
        }
