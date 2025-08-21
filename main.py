import os
import json
from flask import Flask, request, jsonify
import pandas as pd
import psycopg2
from prophet import Prophet # Keep for now, but consider optimization later
from dotenv import load_dotenv
from fetch_aws_cost import get_aws_cost_data # Import the function
from datetime import datetime # Import datetime for date validation

load_dotenv()

app = Flask(__name__)

# Get individual PostgreSQL environment variables
db_host = os.getenv("POSTGRES_DB_HOST")
db_port = os.getenv("POSTGRES_DB_PORT")
db_user = os.getenv("POSTGRES_DB_USER")
db_password = os.getenv("POSTGRES_DB_PASSWORD")
db_database = os.getenv("POSTGRES_DB_DATABASE")

@app.route("/")
def hello_world():
  """Example Hello World route."""
  name = os.environ.get("NAME", "World")
  return f"Hello {name}!"

@app.route("/forecast", methods=['GET'])
def forecast(): # Consider moving Prophet initialization and fitting outside this route for performance
    """Performs time series forecasting by aggregating total daily costs from the details table."""
    """Performs time series forecasting by aggregating total daily costs from the details table."""
    if not all([db_host, db_port, db_user, db_password, db_database]):
        return jsonify({"error": "Database connection variables not configured."}), 500

    conn = None
    try:
        conn = psycopg2.connect(
            host=db_host, port=db_port, user=db_user, password=db_password, database=db_database
        )
        cur = conn.cursor()

        # UPDATED QUERY: Sums the costs per day from your new details table
        cur.execute("""
            SELECT date, SUM(cost) as total_cost
            FROM aws_cost_details
            GROUP BY date
            ORDER BY date
        """)
        data = cur.fetchall()

        if not data:
            return jsonify({"error": "No AWS cost data found in the database."}), 404

        df = pd.DataFrame(data, columns=['ds', 'y'])
        df['ds'] = pd.to_datetime(df['ds'])

        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=30)
        forecast_data = model.predict(future)

        return jsonify(forecast_data[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records'))

    except Exception as e:
        print(f"Error fetching data or performing forecast: {e}")
        return jsonify({"error": "An error occurred during forecasting."}), 500
    finally:
        if conn:
            cur.close()
            conn.close()

@app.route("/fetch_aws_cost", methods=['GET'])
def fetch_aws_cost():
    """Fetches detailed AWS cost data, parses it using context, and stores it in the database."""
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')

    if not start_date_str or not end_date_str:
        return jsonify({"error": "Please provide both start_date and end_date."}), 400

    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d').strftime('%Y-%m-%d')
    except ValueError:
        return jsonify({"error": "Invalid date format. Please use YYYY-MM-DD."}), 400

    cost_data = get_aws_cost_data(start_date, end_date)
    if cost_data is None:
        return jsonify({"error": "Failed to fetch AWS cost data."}), 500

    print(f"Received {len(cost_data)} raw data items from get_aws_cost_data.")

    if not cost_data:
        return jsonify({"message": "No AWS cost data found for the specified date range."}), 200

    if not all([db_host, db_port, db_user, db_password, db_database]):
        return jsonify({"error": "Database connection variables not configured."}), 500

    conn = None
    try:
        conn = psycopg2.connect(
            host=db_host, port=db_port, user=db_user, password=db_password, database=db_database
        )
        cur = conn.cursor()

        # === START: NEW CONTEXT-AWARE PARSING LOGIC ===
        data_to_insert = []
        for result in cost_data:
            date = result.get("TimePeriod", {}).get("Start")
            # The context you added in the fetch function
            group_by_keys = result.get("GroupByKeys")

            # Skip if the data is missing the date or the context
            if not date or not group_by_keys:
                continue

            for group in result.get("Groups", []):
                keys = group.get("Keys", [])
                metrics = group.get("Metrics", {}).get("UnblendedCost", {})
                amount = metrics.get("Amount")
                unit = metrics.get("Unit")

                if amount is None or unit is None:
                    continue

                try:
                    cost_amount = float(amount)
                    
                    # Create a dictionary mapping the dimension name to its value
                    # e.g., {'SERVICE': 'Amazon S3', 'REGION': 'us-east-1'}
                    dimensions_map = dict(zip(group_by_keys, keys))

                    # Safely get each value, providing 'N/A' as a default
                    # to satisfy the NOT NULL constraint in your table.
                    service = dimensions_map.get('SERVICE', 'N/A')
                    region = dimensions_map.get('REGION', 'N/A')
                    usage_type = dimensions_map.get('USAGE_TYPE', 'N/A')
                    operation = dimensions_map.get('OPERATION', 'N/A')
                    linked_account = dimensions_map.get('LINKED_ACCOUNT', 'N/A')
                    
                    data_to_insert.append((
                        date, service, region, usage_type, operation, linked_account, cost_amount, unit
                    ))
                except (ValueError, TypeError):
                    print(f"Skipping item with invalid cost value: {amount}")
                    continue
        # === END: NEW CONTEXT-AWARE PARSING LOGIC ===

        print(f"Processed {len(data_to_insert)} items for insertion into the database.")

        if not data_to_insert:
            return jsonify({"message": "No valid data to insert after processing."}), 200

        # UPDATED INSERT QUERY: Matches the 'aws_cost_details' table structure
        insert_query = """
            INSERT INTO aws_cost_details (date, service, region, usage_type, operation, linked_account, cost, unit)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        """
        cur.executemany(insert_query, data_to_insert)

        conn.commit()
        print(f"Successfully stored/updated {len(data_to_insert)} records in the database.")

        return jsonify({"message": f"Successfully fetched and stored/updated {len(data_to_insert)} AWS cost records."}), 200

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error storing data in database: {e}")
        return jsonify({"error": "An error occurred while storing data."}), 500
    finally:
        if conn:
            cur.close()
            conn.close()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))