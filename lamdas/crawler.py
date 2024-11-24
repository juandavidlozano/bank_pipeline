import boto3

# Initialize Glue client
glue_client = boto3.client('glue')

# Define your Glue Crawler name
CRAWLER_NAME = 'processed-data-crawler'

def lambda_handler(event, context):
    """
    Lambda function triggered by S3 event to start Glue Crawler.
    """
    try:
        # Log the S3 event for debugging
        print("Received event:", event)

        # Start the Glue Crawler
        response = glue_client.start_crawler(Name=CRAWLER_NAME)
        print(f"Crawler '{CRAWLER_NAME}' started successfully.")

        return {
            'statusCode': 200,
            'body': f"Crawler '{CRAWLER_NAME}' started successfully."
        }

    except glue_client.exceptions.CrawlerRunningException:
        print(f"Crawler '{CRAWLER_NAME}' is already running.")
        return {
            'statusCode': 400,
            'body': f"Crawler '{CRAWLER_NAME}' is already running."
        }

    except Exception as e:
        print(f"Error starting crawler: {str(e)}")
        return {
            'statusCode': 500,
            'body': f"Error starting crawler: {str(e)}"
        }
