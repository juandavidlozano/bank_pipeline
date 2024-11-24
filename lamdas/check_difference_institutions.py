import boto3
import urllib.parse
import urllib.request
import json
import logging
from datetime import datetime

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# FDIC Institutions API base URL
INSTITUTIONS_API = "https://banks.data.fdic.gov/api/institutions"

# S3 bucket details
BUCKET_NAME = "alpharank"
INSTITUTIONS_PREFIX = "institutions/"

# Initialize S3 client
s3_client = boto3.client("s3")


def fetch_and_count_rows():
    """Fetch CERT data from the FDIC Institutions API and count the total rows."""
    logger.info("Fetching and counting rows from FDIC API...")
    total_rows = 0
    offset = 0
    page_size = 1000  # Default page size

    while True:
        params = {
            "filters": "ACTIVE:1",  # Filter for active institutions
            "fields": "CERT",  # Fetch only the CERT field
            "format": "json",  # Response format
            "limit": page_size,  # Number of rows per request
            "offset": offset,  # Offset for pagination
        }
        query_string = urllib.parse.urlencode(params)
        url = f"{INSTITUTIONS_API}?{query_string}"

        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())

                # Check if 'data' exists in the response and count rows
                if "data" in data and len(data["data"]) > 0:
                    current_count = len(data["data"])
                    total_rows += current_count
                    offset += page_size
                    logger.info(f"Fetched {current_count} rows, total so far: {total_rows}")
                else:
                    logger.info("No more data to fetch.")
                    break
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            break

    return total_rows


def get_latest_folder(bucket_name, prefix):
    """Get the latest date-time folder in the S3 bucket for a given prefix."""
    logger.info(f"Fetching the latest folder in S3 bucket: {bucket_name}/{prefix}")
    paginator = s3_client.get_paginator("list_objects_v2")
    response_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix, Delimiter="/")

    latest_folder = None
    for page in response_iterator:
        if "CommonPrefixes" in page:
            folders = [prefix["Prefix"] for prefix in page["CommonPrefixes"]]
            latest_folder = max(
                folders, key=lambda f: datetime.strptime(f.rstrip("/").split("/")[-1], "%Y-%m-%d_%H-%M-%S")
            )

    if latest_folder:
        logger.info(f"Latest folder found: {latest_folder}")
    else:
        logger.info("No folders found.")
    return latest_folder


def fetch_total_records_from_s3(bucket_name, folder_prefix):
    """Fetch total records from the total_records.csv file in the latest folder."""
    key = f"{folder_prefix}total_records.csv"
    logger.info(f"Fetching total records from S3 file: s3://{bucket_name}/{key}")
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        content = response["Body"].read().decode("utf-8").strip()
        total_records = int(content.split("\n")[1])  # Assume "total_records\n<number>" format
        logger.info(f"Total records from S3: {total_records}")
        return total_records
    except Exception as e:
        logger.error(f"Error fetching total records from S3: {e}")
        raise


def compare_totals(fetched_total, s3_total):
    """Compare total records and return whether they are different."""
    if fetched_total != s3_total:
        logger.info(f"Different: Fetched total ({fetched_total}) does not match S3 total ({s3_total}).")
        return True  # Records are different
    else:
        logger.info("Not Different: Fetched total matches S3 total.")
        return False  # Records are not different


def lambda_handler(event, context):
    """AWS Lambda handler."""
    try:
        # Step 1: Fetch and count total rows from the API
        fetched_total = fetch_and_count_rows()

        # Step 2: Get the latest folder in the institutions bucket
        latest_folder = get_latest_folder(BUCKET_NAME, INSTITUTIONS_PREFIX)
        if not latest_folder:
            logger.error("No latest folder found in S3. Exiting.")
            return {
                "statusCode": 404,
                "body": json.dumps({"message": "No latest folder found in S3."}),
            }

        # Step 3: Fetch the total records from the S3 bucket
        s3_total = fetch_total_records_from_s3(BUCKET_NAME, latest_folder)

        # Step 4: Compare totals and return result
        records_different = compare_totals(fetched_total, s3_total)

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "records_different": records_different,
                    "fetched_total": fetched_total,
                    "s3_total": s3_total,
                }
            ),
        }

    except Exception as e:
        logger.error(f"Error in process: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
        }
