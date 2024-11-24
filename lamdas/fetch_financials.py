import boto3
import json
import urllib.parse
import urllib.request
from datetime import datetime

# FDIC API base URL for financial data
BASE_URL = "https://banks.data.fdic.gov/api/financials"

# Initialize S3 client
s3_client = boto3.client('s3')

def fetch_data_for_quarter(quarter, limit=1000):
    """Fetch data for a specific quarter from the FDIC API with pagination."""
    print(f"Fetching data for all banks (Quarter: {quarter})...")
    offset = 0
    all_data = []

    while True:
        params = {
            "filters": f'REPDTE:"{quarter}"',
            "fields": "NAME,CHARTER,DEPDOM,ASSET,REPDTE,WEBADDR,CITY,STNAME,CERT",
            "format": "json",
            "limit": limit,
            "offset": offset  # Add offset for pagination
        }
        query_string = urllib.parse.urlencode(params)
        url = f"{BASE_URL}?{query_string}"
        
        try:
            with urllib.request.urlopen(url) as response:
                data = json.loads(response.read().decode())
                if "data" in data and data["data"]:
                    records = [record["data"] for record in data["data"]]
                    all_data.extend(records)
                    offset += limit  # Move to the next batch
                else:
                    print(f"No more data found for quarter {quarter} (offset: {offset}).")
                    break
        except Exception as e:
            print(f"Error fetching data for quarter {quarter} (offset: {offset}): {e}")
            break

    return all_data

def save_to_s3(bucket_name, prefix, data):
    """Save the data to S3 as a CSV file."""
    # Define CSV headers
    headers = ["NAME", "CHARTER", "DEPDOM", "ASSET", "REPDTE", "WEBADDR", "CITY", "STNAME", "CERT"]
    
    # Convert data to CSV format
    csv_lines = [",".join(headers)]  # Add header line
    for record in data:
        csv_lines.append(",".join([str(record.get(header, "")) for header in headers]))
    
    # Combine lines into a CSV string
    csv_content = "\n".join(csv_lines)
    
    # Create S3 key
    s3_key = f"{prefix}/all_banks_extended_last_two_quarters.csv"
    
    # Upload to S3
    s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=csv_content.encode('utf-8'))
    print(f"Data saved to s3://{bucket_name}/{s3_key}")

def lambda_handler(event, context):
    """AWS Lambda handler."""
    try:
        # Read input parameters from the event
        last_quarter = event.get("LAST_QUARTER")
        previous_quarter = event.get("PREVIOUS_QUARTER")
        bucket_name = event.get("BUCKET_NAME", "alpharank")  # Default bucket name

        if not last_quarter or not previous_quarter:
            raise ValueError("LAST_QUARTER and PREVIOUS_QUARTER must be provided in the event.")

        # Fetch data for the last two quarters
        last_quarter_data = fetch_data_for_quarter(last_quarter)
        previous_quarter_data = fetch_data_for_quarter(previous_quarter)

        # Combine the data
        combined_data = last_quarter_data + previous_quarter_data

        if combined_data:
            # Create a folder with the current date-time in the prefix
            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            prefix = f"financials/{current_datetime}"

            # Save the data to S3
            save_to_s3(bucket_name, prefix, combined_data)

            return {
                "statusCode": 200,
                "body": json.dumps({
                    "message": f"Data successfully saved to s3://{bucket_name}/{prefix}/",
                    "rows": len(combined_data)
                })
            }
        else:
            return {
                "statusCode": 200,
                "body": json.dumps({"message": "No data to save."})
            }

    except Exception as e:
        print(f"Error: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
