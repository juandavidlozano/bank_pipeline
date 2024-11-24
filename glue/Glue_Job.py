import sys
import csv
import io
import boto3  # Import boto3 for AWS service interactions
from datetime import datetime
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import udf, trim, col
from pyspark.sql.types import StringType

# AWS Glue Configuration
AWS_REGION = 'us-east-1'
S3_BUCKET_NAME = 'alpharank'  # Replace with your actual bucket name if different
INSTITUTIONS_PREFIX = 'institutions/'
FINANCIALS_PREFIX = 'financials/'
PROCESSED_PREFIX = 'processed_data/'  # Ensure this ends with '/'
OUTPUT_FILE_NAME = 'merged_data.csv'

# Hardcoded job name
JOB_NAME = 'Glue Job'

glueContext = GlueContext(SparkContext.getOrCreate())
spark = glueContext.spark_session

# User-Defined Function to convert date
def convert_date(rep_date):
    """Convert REPDTE into a proper date format."""
    try:
        return datetime.strptime(str(rep_date), '%Y%m%d').strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        return rep_date  # Return the original value if conversion fails

# Register the UDF
convert_date_udf = udf(convert_date, StringType())

# Helper Functions
def get_latest_date_subfolder(s3_client, bucket_name, prefix):
    """Fetch the latest date subfolder from the specified S3 prefix using boto3."""
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix, Delimiter='/')
    if 'CommonPrefixes' not in response:
        raise Exception(f"No subfolders found in S3 bucket '{bucket_name}' with prefix '{prefix}'.")

    subfolders = [common_prefix['Prefix'] for common_prefix in response['CommonPrefixes']]
    sorted_subfolders = sorted(
        subfolders,
        key=lambda x: datetime.strptime(x.strip('/').split('/')[-1], '%Y-%m-%d_%H-%M-%S'),
        reverse=True
    )
    return sorted_subfolders[0]  # Return the latest subfolder

def read_csv_from_s3(bucket, file_path):
    """Read a CSV file from S3 into a Spark DataFrame."""
    s3_path = f"s3://{bucket}/{file_path}"
    return spark.read.csv(s3_path, header=True, inferSchema=True)

def merge_data(financials_df, institutions_df):
    """Perform a left join of financials and institutions data on 'CERT'."""
    from pyspark.sql.functions import col, trim

    # Ensure 'CERT' columns are of the same data type and trim whitespace
    financials_df = financials_df.withColumn('CERT', trim(financials_df['CERT'].cast('string')))
    institutions_df = institutions_df.withColumn('CERT', trim(institutions_df['CERT'].cast('string')))

    # Rename overlapping columns in institutions_df to avoid conflicts, except 'CERT'
    institutions_df = institutions_df.withColumnRenamed('NAME', 'INSTITUTION_NAME')
    institutions_df = institutions_df.withColumnRenamed('CHARTER', 'INSTITUTION_CHARTER')
    institutions_df = institutions_df.withColumnRenamed('WEBADDR', 'INSTITUTION_WEBADDR')
    # Do not rename 'CERT'

    # Perform the left join on 'CERT' column
    merged_df = financials_df.join(
        institutions_df,
        on='CERT',
        how='left'
    )

    # Select and rename columns
    merged_df = merged_df.select(
        col('INSTITUTION_CHARTER').alias('Charter Number'),
        col('INSTITUTION_WEBADDR').alias('Web Domain'),
        col('CITY').alias('City'),
        col('STNAME').alias('State'),
        col('ASSET').alias('Total Assets'),
        col('DEPDOM').alias('Total Deposits'),
        col('REPDTE').alias('Quarter Date'),
        col('NAME').alias('Bank Name'),                  # From financials_df
        col('INSTITUTION_NAME').alias('Institution Name'),  # From institutions_df
        col('CERT')  # Use the 'CERT' column
    )

    # Convert 'Quarter Date' to proper date format
    merged_df = merged_df.withColumn('Quarter Date', convert_date_udf('Quarter Date'))

    # Sort the DataFrame by 'CERT' and 'Quarter Date'
    merged_df = merged_df.orderBy('CERT', 'Quarter Date')

    return merged_df

def write_csv_to_s3(dataframe, bucket, prefix, filename):
    """Write merged data back to S3 with a specific filename."""
    import uuid

    s3 = boto3.resource('s3', region_name=AWS_REGION)
    s3_client = boto3.client('s3', region_name=AWS_REGION)

    # Generate a unique temporary directory path
    temp_dir = f"s3://{bucket}/{prefix}temp_{uuid.uuid4()}/"

    # Write DataFrame to the temporary directory
    dataframe.coalesce(1).write.mode("overwrite").csv(temp_dir, header=True)

    # List the files in the temporary directory
    temp_dir_no_scheme = temp_dir.replace(f"s3://{bucket}/", "")
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=temp_dir_no_scheme)

    # Find the part file
    part_file = None
    for obj in response.get('Contents', []):
        key = obj['Key']
        if key.endswith('.csv'):
            part_file = key
            break

    if not part_file:
        raise Exception("No part file found in the temporary output directory.")

    # Copy the part file to the desired location with the desired filename
    copy_source = {'Bucket': bucket, 'Key': part_file}
    dest_key = f"{prefix}{filename}"
    s3.Object(bucket, dest_key).copy(copy_source)

    # Delete the temporary directory
    delete_temp_directory(s3_client, bucket, temp_dir_no_scheme)

def delete_temp_directory(s3_client, bucket, prefix):
    """Delete all objects under the specified prefix."""
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if 'Contents' in response:
        objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]
        s3_client.delete_objects(Bucket=bucket, Delete={'Objects': objects_to_delete})

# Main Job Function
def main():
    # Initialize the S3 client
    s3_client = boto3.client('s3', region_name=AWS_REGION)

    # Get the latest subfolder for institutions
    latest_institutions_folder = get_latest_date_subfolder(s3_client, S3_BUCKET_NAME, INSTITUTIONS_PREFIX)
    institutions_file_path = f"{latest_institutions_folder}institutions_data.csv"
    institutions_df = read_csv_from_s3(S3_BUCKET_NAME, institutions_file_path)

    # Get the latest subfolder for financials
    latest_financials_folder = get_latest_date_subfolder(s3_client, S3_BUCKET_NAME, FINANCIALS_PREFIX)
    financials_file_path = f"{latest_financials_folder}all_banks_extended_last_two_quarters.csv"
    financials_df = read_csv_from_s3(S3_BUCKET_NAME, financials_file_path)

    # Merge the data
    merged_df = merge_data(financials_df, institutions_df)

    # Write merged data back to S3
    write_csv_to_s3(merged_df, S3_BUCKET_NAME, PROCESSED_PREFIX, OUTPUT_FILE_NAME)

# Glue Job Entry Point
if __name__ == "__main__":
    job = Job(glueContext)
    job.init(JOB_NAME, {})
    main()
    job.commit()
