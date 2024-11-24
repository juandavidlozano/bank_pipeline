import boto3
import json
import logging
import urllib.parse
import urllib.request
from datetime import datetime

# Set up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# FDIC Institutions API base URL
INSTITUTIONS_API = "https://banks.data.fdic.gov/api/institutions"

# S3 bucket details
BUCKET_NAME = "alpharank"
FOLDER_NAME = "institutions"

# Initialize S3 client
s3_client = boto3.client('s3')

def fetch_all_institution_data():
    """Fetch all data from the FDIC Institutions API by handling pagination."""
    logger.info("Fetching data for institutions without limit...")
    all_data = []
    offset = 0
    page_size = 1000  # Default page size
    
    while True:
        params = {
            "filters": "ACTIVE:1",  # Filter for active institutions
            "fields": "NAME,CHARTER,WEBADDR,NAMEHCR,CERT",  # Fields to fetch
            "format": "json",  # Response format
            "limit": page_size,  # Number of rows per request
            "offset": offset  # Offset for pagination
        }
        query_string = urllib.parse.urlencode(params)
        url = f"{INSTITUTIONS_API}?{query_string}"
        
        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
                if "data" in data and len(data["data"]) > 0:
                    all_data.extend([record["data"] for record in data["data"]])
                    offset += page_size
                    logger.info(f"Fetched {len(data['data'])} records, total so far: {len(all_data)}")
                else:
                    logger.info("No more data to fetch.")
                    break
        except Exception as e:
            logger.error(f"Error fetching institution data: {e}")
            break

    return all_data

def save_to_s3(bucket_name, prefix, filename, content):
    """Save content to S3."""
    s3_key = f"{prefix}/{filename}"
    s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=content.encode('utf-8'))
    logger.info(f"Saved {filename} to s3://{bucket_name}/{s3_key}")

def lambda_handler(event, context):
    """AWS Lambda handler."""
    try:
        # Fetch all institution data
        institution_data = fetch_all_institution_data()

        if institution_data:
            # Create a folder with the current date-time in the prefix
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            prefix = f"{FOLDER_NAME}/{current_datetime}"

            # Save institution data as a CSV
            headers = ["NAME", "CHARTER", "WEBADDR", "NAMEHCR", "CERT"]
            csv_lines = [",".join(headers)]  # Add header line
            for record in institution_data:
                csv_lines.append(",".join([str(record.get(header, "")) for header in headers]))
            csv_content = "\n".join(csv_lines)
            save_to_s3(BUCKET_NAME, prefix, "institutions_data.csv", csv_content)

            # Save total records as a separate CSV
            total_records_content = f"total_records\n{len(institution_data)}"
            save_to_s3(BUCKET_NAME, prefix, "total_records.csv", total_records_content)

            return {
                "statusCode": 200,
                "body": json.dumps({
                    "message": f"Data successfully saved to s3://{BUCKET_NAME}/{prefix}/",
                    "total_records": len(institution_data)
                })
            }
        else:
            logger.info("No data to save.")
            return {
                "statusCode": 200,
                "body": json.dumps({"message": "No data to save."})
            }

    except Exception as e:
        logger.error(f"Error: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
