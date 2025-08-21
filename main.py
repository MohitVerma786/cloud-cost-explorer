import os
import threading
import time
from datetime import datetime, timedelta

import pandas as pd
import psycopg2
from dateutil.relativedelta import relativedelta
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
cached_data = {}
cache_lock = threading.Lock()
GRANULARITIES = ['day', 'month', 'year']
RETRAIN_INTERVAL_SECONDS = 86400  # Retrain once every 24 hours

def fetch_detailed_cost_data() -> pd.DataFrame:
    """Fetches raw, unaggregated cost data from the database."""
    sql_query = """
        SELECT date, cost, service, region, usage_type, operation, linked_account
        FROM aws_cost_details
        ORDER BY date;
    """
    conn = None
    try:
        conn = conn_pool.getconn()
        df = pd.read_sql_query(sql_query, conn)
        if df.empty:
            return pd.DataFrame()
        
        df['date'] = pd.to_datetime(df['date'])
        df['cost'] = df['cost'].astype(float)
        return df
    finally:
        if conn:
            conn_pool.putconn(conn)

def retrain_models():
    """A background task that fetches detailed data and trains models for all granularities."""
    print("Background training task started.")
    while True:
        try:
            print("Fetching detailed data for training cycle...")
            detailed_df = fetch_detailed_cost_data()

            if detailed_df.empty:
                print("No data found in database. Skipping training cycle.")
                time.sleep(RETRAIN_INTERVAL_SECONDS)
                continue

            for granularity in GRANULARITIES:
                print(f"[{granularity}] Aggregating data for training...")
                
                if granularity == 'day':
                    agg_df = detailed_df.groupby(detailed_df['date'].dt.date)['cost'].sum().reset_index()
                else:
                    freq = 'M' if granularity == 'month' else 'Y'
                    agg_df = detailed_df.groupby(pd.Grouper(key='date', freq=freq))['cost'].sum().reset_index()

                agg_df.rename(columns={'date': 'ds', 'cost': 'y'}, inplace=True)
                
                if len(agg_df) < 2:
                    print(f"[{granularity}] Not enough aggregated data points to train model. Skipping.")
                    continue

                holidays_df = None
                if granularity == 'day' and not agg_df.empty:
                    future_period = timedelta(days=365 * 5)
                    date_range = pd.date_range(start=agg_df['ds'].min(), end=agg_df['ds'].max() + future_period)
                    billing_days = date_range[date_range.day == 1]
                    holidays_df = pd.DataFrame({
                        'holiday': 'monthly_billing',
                        'ds': pd.to_datetime(billing_days),
                        'lower_window': 0, 'upper_window': 0,
                    })

                print(f"[{granularity}] Fitting new model with custom holidays...")
                model = Prophet(holidays=holidays_df, interval_width=0.99)
                model.fit(agg_df)

                with cache_lock:
                    cached_data[granularity] = {
                        "model": model,
                        "details": detailed_df if granularity == 'day' else None
                    }
                print(f"[{granularity}] Model updated successfully.")
        except Exception as e:
            print(f"An error occurred during the retraining cycle: {e}")
        
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
    Performs time series forecasting and provides detailed anomaly detection.
    """
    granularity = request.args.get('granularity', 'day').lower()
    if granularity not in GRANULARITIES:
        return jsonify({"error": "Invalid granularity. Choose 'day', 'month', or 'year'."}), 400

    with cache_lock:
        model_data = cached_data.get(granularity)
        detailed_data = cached_data.get('day', {}).get('details')

    if not model_data or not hasattr(detailed_data, 'empty'):
        return jsonify({"error": f"The model or data for '{granularity}' granularity is not ready yet."}), 503

    model = model_data["model"]

    try:
        periods, freq = (12, 'M') if granularity == 'month' else (3, 'Y') if granularity == 'year' else (30, 'D')
        
        future = model.make_future_dataframe(periods=periods, freq=freq)
        full_forecast_df = model.predict(future)
        
        results_df = pd.merge(full_forecast_df, model.history, on='ds', how='left')
        
        anomalies_df = results_df[
            (results_df['y'] > results_df['yhat_upper']) | 
            (results_df['y'] < results_df['yhat_lower'])
        ]
        
        anomalies_list = []
        if not anomalies_df.empty:
            if granularity == 'day':
                anomalous_dates = anomalies_df['ds'].dt.date
                detailed_anomalies = detailed_data[detailed_data['date'].dt.date.isin(anomalous_dates)]
                
                # --- NEW: Filter out zero-cost items from the anomaly report ---
                meaningful_anomalies = detailed_anomalies[detailed_anomalies['cost'] > 0.0]
                anomalies_list = meaningful_anomalies.to_dict(orient='records')
                # --- END of new logic ---

            else:
                for _, row in anomalies_df.iterrows():
                    start_date = row['ds']
                    end_date = (start_date + relativedelta(months=1) - timedelta(days=1)) if granularity == 'month' else (start_date + relativedelta(years=1) - timedelta(days=1))
                    
                    period_details = detailed_data[detailed_data['date'].between(start_date, end_date)]
                    
                    # --- NEW: Filter out zero-cost items here as well ---
                    meaningful_details = period_details[period_details['cost'] > 0.0]
                    # --- END of new logic ---

                    anomalies_list.append({
                        "granularity": granularity,
                        "startDate": start_date.strftime('%Y-%m-%d'),
                        "endDate": end_date.strftime('%Y-%m-%d'),
                        "totalAnomalousCost": row['y'],
                        "costDetails": meaningful_details.to_dict(orient='records')
                    })

        full_forecast_df.rename(columns={'ds': 'date', 'yhat': 'predictedCost', 'yhat_lower': 'minExpectedCost', 'yhat_upper': 'maxExpectedCost'}, inplace=True)
        cost_columns = ['predictedCost', 'minExpectedCost', 'maxExpectedCost']
        full_forecast_df[cost_columns] = full_forecast_df[cost_columns].clip(lower=0)
        future_forecast_df = full_forecast_df[full_forecast_df['date'].dt.date >= datetime.now().date()]
        response_data = future_forecast_df[['date', 'predictedCost', 'minExpectedCost', 'maxExpectedCost']].to_dict(orient='records')
        
        for record in response_data: record['date'] = record['date'].strftime('%Y-%m-%d')
        for anomaly in anomalies_list:
            if 'date' in anomaly: anomaly['date'] = anomaly['date'].strftime('%Y-%m-%d %H:%M:%S')
            if 'costDetails' in anomaly:
                for detail in anomaly['costDetails']: detail['date'] = detail['date'].strftime('%Y-%m-%d %H:%M:%S')

        return jsonify({"forecast": response_data, "anomalies": anomalies_list})
    except Exception as e:
        print(f"Error during prediction for '{granularity}': {e}")
        return jsonify({"error": "An error occurred during forecasting."}), 500

# The /fetch_aws_cost route remains unchanged...
@app.route("/fetch_aws_cost", methods=['GET'])
def fetch_aws_cost():
    """Fetches detailed AWS cost data, parses it, and stores it in the database."""
    start_date_str = request.args.get('start_date')
    end_date_str = request.args.get('end_date')

    if not start_date_str or not end_date_str:
        return jsonify({"error": "Please provide both start_date and end_date."}), 400

    try:
        datetime.strptime(start_date_str, '%Y-%m-%d')
        datetime.strptime(end_date_str, '%Y-%m-%d')
    except ValueError:
        return jsonify({"error": "Invalid date format. Please use YYYY-MM-DD."}), 400

    cost_data = get_aws_cost_data(start_date_str, end_date_str)
    if cost_data is None:
        return jsonify({"error": "Failed to fetch AWS cost data."}), 500

    if not cost_data:
        return jsonify({"message": "No AWS cost data found for the specified date range."}), 200

    conn = None
    try:
        conn = psycopg2.connect(
            host=db_host, port=db_port, user=db_user, password=db_password, database=db_database
        )
        cur = conn.cursor()

        data_to_insert = []
        for result in cost_data:
            date = result.get("TimePeriod", {}).get("Start")
            group_by_keys = result.get("GroupByKeys")

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
                    
                    data_to_insert.append((
                        date,
                        dimensions_map.get('SERVICE', 'N/A'),
                        dimensions_map.get('REGION', 'N/A'),
                        dimensions_map.get('USAGE_TYPE', 'N/A'),
                        dimensions_map.get('OPERATION', 'N/A'),
                        dimensions_map.get('LINKED_ACCOUNT', 'N/A'),
                        cost_amount,
                        unit
                    ))
                except (ValueError, TypeError):
                    continue
        
        if not data_to_insert:
            return jsonify({"message": "No valid data to insert after processing."}), 200

        insert_query = """
            INSERT INTO aws_cost_details (date, service, region, usage_type, operation, linked_account, cost, unit)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        cur.executemany(insert_query, data_to_insert)
        conn.commit()

        return jsonify({"message": f"Successfully fetched and stored {len(data_to_insert)} AWS cost records."}), 200

    except Exception as e:
        if conn: conn.rollback()
        print(f"Error storing data in database: {e}")
        return jsonify({"error": "An error occurred while storing data."}), 500
    finally:
        if conn:
            cur.close()
            conn.close()


if __name__ == "__main__":
    retrain_thread = threading.Thread(target=retrain_models, daemon=True)
    retrain_thread.start()
    
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))