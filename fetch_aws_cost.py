import os
import boto3
import psycopg2
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get AWS credentials and database connection string from environment variables
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")

db_host = os.getenv("POSTGRES_DB_HOST")
db_port = os.getenv("POSTGRES_DB_PORT")
db_user = os.getenv("POSTGRES_DB_USER")
db_password = os.getenv("POSTGRES_DB_PASSWORD")
db_database = os.getenv("POSTGRES_DB_DATABASE")

if not all([aws_access_key_id, aws_secret_access_key, db_host, db_port, db_user, db_password, db_database]):
    print("Error: Missing required environment variables.")
    exit(1)

# Configure AWS Cost Explorer client
try:
    ce_client = boto3.client(
        "ce",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
except Exception as e:
    print(f"Error creating AWS Cost Explorer client: {e}")
    exit()

# Define the date range for cost data (last day)
end_date = datetime.today().strftime("%Y-%m-%d")
start_date = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

# Function to fetch cost data
# In fetch_aws_cost.py

# Function to fetch cost data
def get_aws_cost_data(start, end):
    all_results = []
    # This list of combinations remains the same
    group_by_combinations = [
        [{"Type": "DIMENSION", "Key": "SERVICE"}],
        [{"Type": "DIMENSION", "Key": "SERVICE"}, {"Type": "DIMENSION", "Key": "REGION"}],
        [{"Type": "DIMENSION", "Key": "SERVICE"}, {"Type": "DIMENSION", "Key": "USAGE_TYPE"}],
        [{"Type": "DIMENSION", "Key": "SERVICE"}, {"Type": "DIMENSION", "Key": "OPERATION"}],
        [{"Type": "DIMENSION", "Key": "SERVICE"}, {"Type": "DIMENSION", "Key": "LINKED_ACCOUNT"}],
    ]

    try:
        for group_by in group_by_combinations:
            print(f"Fetching data grouped by: {[g['Key'] for g in group_by]}")
            # --- START: MODIFIED LOGIC ---
            # Get the dimension keys for the current API call to use as context
            group_by_keys = [g['Key'] for g in group_by]
            
            next_page_token = None
            while True:
                params = {
                    "TimePeriod": {"Start": start, "End": end},
                    "Granularity": "DAILY",
                    "Metrics": ["UnblendedCost"],
                    "GroupBy": group_by,
                }
                if next_page_token:
                    params["NextPageToken"] = next_page_token
                
                try:
                    response = ce_client.get_cost_and_usage(**params)
                    
                    print(f"  Received response for group: {group_by_keys}")
                    # Get the results for this page
                    results_from_api = response.get("ResultsByTime", [])
                    
                    # Add the 'GroupByKeys' context to each result object
                    for result in results_from_api:
                        result['GroupByKeys'] = group_by_keys
                    
                    # Add the enriched results to the main list
                    all_results.extend(results_from_api)

                    print(f"  Processed {len(results_from_api)} results. Total collected: {len(all_results)}")
                    # --- END: MODIFIED LOGIC ---
                    
                    next_page_token = response.get("NextPageToken")
                    if not next_page_token:
                        break

                except Exception as e:
                    print(f"  API call failed with error: {e}")
                    print(f"Error fetching AWS cost data for group: {group_by_keys} - {e}")
                    break

    except Exception as e:
        print(f"An error occurred during the API calls: {e}")
        return None
        
    return all_results

# Function to store data in PostgreSQL
def store_cost_data(data_to_insert):
    conn = None
    try:
        # Get individual PostgreSQL environment variables from environment variables
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            database=db_database)
        cur = conn.cursor()

        insert_query = """
            INSERT INTO aws_cost_details (date, service, region, usage_type, operation, linked_account, cost, unit)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT ON CONSTRAINT unique_cost_entry DO UPDATE SET
                cost = EXCLUDED.cost,
                unit = EXCLUDED.unit;
        """
        cur.executemany(insert_query, data_to_insert)

        conn.commit()
        print(f"Successfully stored/updated {len(data_to_insert)} records in the database.")

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error storing data in database: {e}")
        raise # Re-raise the exception after logging
    finally:
        if conn:
            cur.close()
            conn.close()

