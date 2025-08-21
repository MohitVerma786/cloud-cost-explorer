import os
import threading
import time
from datetime import datetime, timedelta

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from prophet import Prophet
from psycopg2.pool import SimpleConnectionPool

from fetch_aws_cost import get_aws_cost_data  # Import the function

load_dotenv()

app = Flask(__name__)

# --- Database Configuration & Connection Pooling ---
db_host = os.getenv("POSTGRES_DB_HOST")
db_port = os.getenv("POSTGRES_DB_PORT")
db_user = os.getenv("POSTGRES_DB_USER")
db_password = os.getenv("POSTGRES_DB_PASSWORD")
db_database = os.getenv("POSTGRES_DB_DATABASE")

# Check for DB config at startup to fail fast
if not all([db_host, db_port, db_user, db_password, db_database]):
    raise ValueError("Database connection variables are not fully configured.")

# Create a single connection pool when the app starts
conn_pool = SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    host=db_host,
    port=db_port,
    user=db_user,
    password=db_password,
    database=db_database
)

# --- Model Caching & Background Training Infrastructure ---
trained_models = {}
model_lock = threading.Lock()
GRANULARITIES = ['day', 'month', 'year']
RETRAIN_INTERVAL_SECONDS = 86400  # Retrain once every 24 hours

def fetch_cost_data(granularity: str) -> pd.DataFrame:
    """Fetches and processes cost data from the database for a given granularity."""
    date_trunc_map = {
        'day': "date",
        'month': "DATE_TRUNC('month', date AT TIME ZONE 'America/Los_Angeles')",
        'year': "DATE_TRUNC('year', date AT TIME ZONE 'America/Los_Angeles')"
    }
    date_grouping = date_trunc_map[granularity]

    sql_query = f"""
        SELECT {date_grouping}::date as ds, SUM(cost) as y
        FROM aws_cost_details
        GROUP BY ds
        ORDER BY ds;
    """
    conn = None
    try:
        conn = conn_pool.getconn()
        with conn.cursor() as cur:
            cur.execute(sql_query)
            data = cur.fetchall()
            if not data:
                return pd.DataFrame(columns=['ds', 'y'])
            
            df = pd.DataFrame(data, columns=['ds', 'y'])
            df['ds'] = pd.to_datetime(df['ds'])
            df['y'] = df['y'].astype(float)
            return df
    finally:
        if conn:
            conn_pool.putconn(conn)

def retrain_models():
    """A background task to periodically retrain models for all granularities."""
    print("Background training task started.")
    while True:
        for granularity in GRANULARITIES:
            try:
                print(f"[{granularity}] Fetching data for retraining...")
                df = fetch_cost_data(granularity)

                if len(df) < 2:
                    print(f"[{granularity}] Not enough data to train model. Skipping.")
                    continue

                print(f"[{granularity}] Fitting new model...")
                model = Prophet()
                model.fit(df)

                with model_lock:
                    trained_models[granularity] = model
                print(f"[{granularity}] Model updated successfully.")
            except Exception as e:
                print(f"Error during background model retraining for '{granularity}': {e}")
        
        print(f"Next model retraining in {timedelta(seconds=RETRAIN_INTERVAL_SECONDS)}.")
        time.sleep(RETRAIN_INTERVAL_SECONDS)


# --- API Routes ---

@app.route("/")
def hello_world():
  """Example Hello World route."""
  name = os.environ.get("NAME", "World")
  return f"Hello {name}!"

@app.route("/forecast", methods=['GET'])
def forecast():
    """
    Performs time series forecasting and anomaly detection using a pre-trained, cached model.
    Anomalies are identified as historical points where the actual cost falls outside
    the model's predicted confidence interval (yhat_upper/yhat_lower).
    """
    granularity = request.args.get('granularity', 'day').lower()
    if granularity not in GRANULARITIES:
        return jsonify({"error": "Invalid granularity. Choose 'day', 'month', or 'year'."}), 400

    with model_lock:
        model = trained_models.get(granularity)

    if not model:
        return jsonify({"error": f"The model for '{granularity}' granularity is not ready. Please try again later."}), 503

    try:
        # Prediction is fast and done on-demand
        future = model.make_future_dataframe(periods=30, freq='D')
        forecast_df = model.predict(future)
        
        # --- NEW: Anomaly detection using Prophet's model internals ---
        # Merge historical actuals (y) with forecasts (yhat, yhat_lower, yhat_upper)
        results_df = pd.merge(forecast_df, model.history, on='ds', how='left')
        
        # Identify points where the actual value 'y' is outside the predicted confidence interval
        anomalies_df = results_df[
            (results_df['y'] > results_df['yhat_upper']) | 
            (results_df['y'] < results_df['yhat_lower'])
        ]
        
        # Format anomalies for the JSON response
        anomalies_list = anomalies_df[['ds', 'y']].to_dict(orient='records')
        for anomaly in anomalies_list:
            anomaly['ds'] = anomaly['ds'].strftime('%Y-%m-%d')
        # --- END of new anomaly logic ---

        response_data = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records')
        
        for record in response_data:
            record['ds'] = record['ds'].strftime('%Y-%m-%d')

        return jsonify({"forecast": response_data, "anomalies": anomalies_list})
    except Exception as e:
        print(f"Error during prediction for '{granularity}': {e}")
        return jsonify({"error": "An error occurred during forecasting."}), 500

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
                    
                    dimensions_map = dict(zip(group_by_keys, keys))

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

        insert_query = """
            INSERT INTO aws_cost_details (date, service, region, usage_type, operation, linked_account, cost, unit)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
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
    # Start the background retraining thread as a daemon so it exits with the main app
    retrain_thread = threading.Thread(target=retrain_models, daemon=True)
    retrain_thread.start()
    
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))