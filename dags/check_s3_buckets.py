from airflow import DAG
from airflow.operators.python_operator import PythonOperator, BranchPythonOperator
from airflow.operators.dummy_operator import DummyOperator
from datetime import datetime
import boto3
import logging
import json
import time

# Specify AWS region
AWS_REGION = 'us-east-1'

# Logging setup
logger = logging.getLogger("airflow.task")
logger.setLevel(logging.INFO)

# Common function to check if an S3 bucket folder is empty
def is_bucket_empty(bucket_name, prefix):
    """Check if an S3 bucket folder is empty."""
    client = boto3.client('s3', region_name=AWS_REGION)
    response = client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    if 'Contents' in response:
        non_empty_objects = [obj for obj in response['Contents'] if obj['Size'] > 0]
        return len(non_empty_objects) == 0  # True if no non-empty objects are found
    return True  # True if 'Contents' is not in the response

# Common function to invoke a Lambda function
def invoke_lambda_function(function_name, payload):
    """Invoke a Lambda function and return its response."""
    client = boto3.client('lambda', region_name=AWS_REGION)
    try:
        response = client.invoke(
            FunctionName=function_name,
            InvocationType='RequestResponse',
            Payload=json.dumps(payload),
        )
        response_payload = json.load(response['Payload'])
        logger.info(f"Lambda {function_name} invoked successfully. Response: {response_payload}")
        return response_payload
    except Exception as e:
        logger.error(f"Error invoking Lambda {function_name}: {e}")
        raise

# Function to run the AWS Glue job
def run_glue_job(**kwargs):
    glue_client = boto3.client('glue', region_name=AWS_REGION)
    job_name = 'Glue Job'  # Replace with your actual Glue job name if different
    try:
        response = glue_client.start_job_run(JobName=job_name)
        job_run_id = response['JobRunId']
        logger.info(f"Glue job '{job_name}' started with JobRunId: {job_run_id}")

        # Wait for the job to complete
        while True:
            status_response = glue_client.get_job_run(JobName=job_name, RunId=job_run_id)
            job_run_state = status_response['JobRun']['JobRunState']
            if job_run_state == 'SUCCEEDED':
                logger.info(f"Glue job '{job_name}' completed successfully.")
                break
            elif job_run_state in ['FAILED', 'STOPPED', 'TIMEOUT']:
                logger.error(f"Glue job '{job_name}' failed with state '{job_run_state}'.")
                raise Exception(f"Glue job '{job_name}' failed with state '{job_run_state}'.")
            else:
                logger.info(f"Glue job '{job_name}' is in state '{job_run_state}'. Waiting for completion...")
                time.sleep(30)  # Wait for 30 seconds before checking again
    except Exception as e:
        logger.error(f"Error running Glue job '{job_name}': {e}")
        raise

# DAG definition
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
}

with DAG(
    'combined_financials_institutions_dag',
    default_args=default_args,
    description='Combined DAG for financials and institutions',
    schedule_interval=None,
    start_date=datetime(2023, 11, 23),
    catchup=False,
) as dag:

    # Start task
    start_task = DummyOperator(
        task_id='start',
    )

    ###########################################
    # Financials Branch
    ###########################################

    # Branch task for financials bucket
    def branch_financials_bucket(**kwargs):
        if is_bucket_empty('alpharank', 'financials/'):
            return 'fetch_last_quarter_variable'  # Proceed directly to fetch
        else:
            return 'check_difference_financials'  # Check differences

    branch_financials = BranchPythonOperator(
        task_id='branch_financials',
        python_callable=branch_financials_bucket,
    )

    # Fetch last and previous quarters
    def fetch_last_quarter_variable(**kwargs):
        from datetime import datetime

        today = datetime.today()

        # Map quarters to their end dates
        quarter_end_dates = {
            1: (3, 31),   # Q1 ends on March 31
            2: (6, 30),   # Q2 ends on June 30
            3: (9, 30),   # Q3 ends on September 30
            4: (12, 31),  # Q4 ends on December 31
        }

        # Determine the current quarter
        if today.month <= 3:
            current_quarter = 1
        elif today.month <= 6:
            current_quarter = 2
        elif today.month <= 9:
            current_quarter = 3
        else:
            current_quarter = 4

        # Adjust for data availability delay (two quarters behind)
        if current_quarter >= 3:
            last_quarter = current_quarter - 2
            last_quarter_year = today.year
        else:
            last_quarter = current_quarter - 2 + 4  # Adjust for year rollover
            last_quarter_year = today.year - 1

        # Calculate the previous quarter
        if last_quarter > 1:
            previous_quarter = last_quarter - 1
            previous_quarter_year = last_quarter_year
        else:
            previous_quarter = 4
            previous_quarter_year = last_quarter_year - 1

        # Get the end dates for the quarters
        last_quarter_end_month, last_quarter_end_day = quarter_end_dates[last_quarter]
        previous_quarter_end_month, previous_quarter_end_day = quarter_end_dates[previous_quarter]

        # Format the dates as YYYYMMDD
        last_quarter_date = f"{last_quarter_year}{last_quarter_end_month:02}{last_quarter_end_day:02}"
        previous_quarter_date = f"{previous_quarter_year}{previous_quarter_end_month:02}{previous_quarter_end_day:02}"

        # Push the values to XCom
        kwargs['ti'].xcom_push(key='last_quarter', value=last_quarter_date)
        kwargs['ti'].xcom_push(key='previous_quarter', value=previous_quarter_date)
        logger.info(f"Last quarter: {last_quarter_date}, Previous quarter: {previous_quarter_date}")

    fetch_quarters = PythonOperator(
        task_id='fetch_last_quarter_variable',
        python_callable=fetch_last_quarter_variable,
    )

    # Check differences for financials
    def check_difference_financials(**kwargs):
        """Invoke check_difference_financials Lambda."""
        response_payload = invoke_lambda_function("check_difference_financials", {})
        logger.info(f"Lambda response: {response_payload}")
        run_needed = response_payload.get('run_needed', False)
        # Push 'run_needed' to XCom
        kwargs['ti'].xcom_push(key='run_needed', value=run_needed)

    check_financials = PythonOperator(
        task_id='check_difference_financials',
        python_callable=check_difference_financials,
    )

    # Decide whether to proceed based on 'run_needed'
    def decide_whether_to_run_fetch_financials(**kwargs):
        run_needed = kwargs['ti'].xcom_pull(task_ids='check_difference_financials', key='run_needed')
        if run_needed:
            return 'invoke_fetch_financials'
        else:
            return 'stop_task_financials'

    decide_task_financials = BranchPythonOperator(
        task_id='decide_whether_to_run_fetch_financials',
        python_callable=decide_whether_to_run_fetch_financials,
    )

    # Task to invoke the fetch financials Lambda
    def run_fetch_financials_lambda(**kwargs):
        last_quarter = kwargs['ti'].xcom_pull(task_ids='fetch_last_quarter_variable', key='last_quarter')
        previous_quarter = kwargs['ti'].xcom_pull(task_ids='fetch_last_quarter_variable', key='previous_quarter')
        payload = {"LAST_QUARTER": last_quarter, "PREVIOUS_QUARTER": previous_quarter, "BUCKET_NAME": "alpharank"}
        invoke_lambda_function("fetch_fiinancials", payload)  # Corrected Lambda function name
        logger.info("Fetch financials Lambda invoked successfully.")

    invoke_fetch_financials = PythonOperator(
        task_id='invoke_fetch_financials',
        python_callable=run_fetch_financials_lambda,
        trigger_rule='none_failed_or_skipped',
    )

    # Dummy task to stop the pipeline
    stop_task_financials = DummyOperator(
        task_id='stop_task_financials',
    )

    # Define dependencies for financials
    start_task >> branch_financials
    branch_financials >> [fetch_quarters, check_financials]
    fetch_quarters >> invoke_fetch_financials
    check_financials >> decide_task_financials
    decide_task_financials >> [invoke_fetch_financials, stop_task_financials]

    ###########################################
    # Institutions Branch
    ###########################################

    # Branch task for institutions bucket
    def branch_institutions_bucket(**kwargs):
        if is_bucket_empty('alpharank', 'institutions/'):
            return 'invoke_fetch_institutions'  # Proceed directly to fetch
        else:
            return 'check_difference_institutions'  # Check differences

    branch_institutions = BranchPythonOperator(
        task_id='branch_institutions',
        python_callable=branch_institutions_bucket,
    )

    # Check differences for institutions
    def check_difference_institutions(**kwargs):
        """Invoke check_difference_institutions Lambda and push result to XCom."""
        response = invoke_lambda_function('check_difference_institutions', {})
        run_needed = response.get('run_needed', False)
        # Push 'run_needed' to XCom
        kwargs['ti'].xcom_push(key='run_needed', value=run_needed)
        logger.info(f"'run_needed' set to {run_needed}")

    check_institutions = PythonOperator(
        task_id='check_difference_institutions',
        python_callable=check_difference_institutions,
    )

    # Decide whether to invoke fetch
    def decide_whether_to_run_fetch_institutions(**kwargs):
        """Decide to proceed based on 'run_needed' from XCom."""
        run_needed = kwargs['ti'].xcom_pull(task_ids='check_difference_institutions', key='run_needed')
        if run_needed:
            return 'invoke_fetch_institutions'
        else:
            return 'stop_task_institutions'

    decide_to_run_fetch_institutions = BranchPythonOperator(
        task_id='decide_whether_to_run_fetch_institutions',
        python_callable=decide_whether_to_run_fetch_institutions,
    )

    stop_task_institutions = DummyOperator(task_id='stop_task_institutions')

    # Invoke fetch institutions Lambda
    def run_fetch_institutions_lambda(**kwargs):
        payload = {}
        response = invoke_lambda_function("fetch_institutions", payload)
        total_records = response.get('total_records', 0)
        kwargs['ti'].xcom_push(key='total_records', value=total_records)
        logger.info("Fetch institutions Lambda invoked successfully.")

    invoke_fetch_institutions = PythonOperator(
        task_id='invoke_fetch_institutions',
        python_callable=run_fetch_institutions_lambda,
        trigger_rule='none_failed_or_skipped',
    )

    write_total_records = PythonOperator(
        task_id='write_total_records',
        python_callable=lambda **kwargs: logger.info("Total records written."),
    )

    # Define dependencies for institutions
    start_task >> branch_institutions
    branch_institutions >> [invoke_fetch_institutions, check_institutions]
    check_institutions >> decide_to_run_fetch_institutions
    decide_to_run_fetch_institutions >> [invoke_fetch_institutions, stop_task_institutions]
    invoke_fetch_institutions >> write_total_records

    ###########################################
    # New Glue Job Task
    ###########################################

    # Task to run the Glue job after both branches have completed
    run_glue_job_task = PythonOperator(
        task_id='run_glue_job',
        python_callable=run_glue_job,
        trigger_rule='none_failed_or_skipped',  # Ensure it runs when both upstream tasks have succeeded
    )

    # Set dependencies: run_glue_job_task depends on both invoke_fetch_financials and write_total_records
    [invoke_fetch_financials, write_total_records] >> run_glue_job_task

    # Optionally, define an end task if needed
    end_task = DummyOperator(
        task_id='end',
    )

    run_glue_job_task >> end_task
