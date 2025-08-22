# main.py
import os
import threading
import time
from datetime import datetime, timedelta
import json
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from sqlalchemy import create_engine, text

from ml_models import TransformerAutoencoder

load_dotenv()

app = Flask(__name__)

# --- Configuration ---
db_host = os.getenv("POSTGRES_DB_HOST")
db_port = os.getenv("POSTGRES_DB_PORT")
db_user = os.getenv("POSTGRES_DB_USER")
db_password = os.getenv("POSTGRES_DB_PASSWORD")
db_database = os.getenv("POSTGRES_DB_DATABASE")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not all([db_host, db_port, db_user, db_password, db_database]):
    raise ValueError("Database connection variables are not fully configured.")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_database}"
engine = create_engine(db_url, pool_size=10, max_overflow=20)

# --- Model Infrastructure ---
model_cache = {}  # Cache for trained model instances
cache_lock = threading.Lock()
GRANULARITIES = ["day", "month", "year"]
RETRAIN_INTERVAL_SECONDS = 86400  # 24 hours
SEQUENCE_LENGTH = 30  # Define sequence length for the model

# --- Data Fetching ---
def fetch_detailed_cost_data() -> pd.DataFrame:
    """Fetches raw, unaggregated cost data from the database."""
    # ... (No changes needed, code is correct)
    sql_query = """
        SELECT date, cost, service, region, usage_type, operation, linked_account
        FROM aws_cost_details
        ORDER BY date;
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql_query), conn)
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df["cost"] = df["cost"].astype(float)
    return df

# --- Background Model Training ---
def retrain_models():
    """A background task that fetches data and trains models for all granularities."""
    print("Background training task started.")
    # Log start of the background process
    print(f"[{datetime.now()}] Background training task started.")
    while True:
        try:
            # Log data fetching start
            print("Fetching detailed data for training cycle...")
            detailed_df = fetch_detailed_cost_data()

            if detailed_df.empty:
                # Log no data found
                print("No data found in database. Skipping training cycle.")
                time.sleep(RETRAIN_INTERVAL_SECONDS)
                continue
            # Log data fetching success
            print(f"[{datetime.now()}] Data fetching complete. Rows fetched: {len(detailed_df)}")

            for granularity in GRANULARITIES:
                print(f"--- Training model for [{granularity}] granularity ---")

                # Aggregate data based on granularity
                # Log aggregation start
                if granularity == "day":
                    agg_df = detailed_df.groupby(detailed_df["date"].dt.date)["cost"].sum().reset_index()
                else:
                    freq = "M" if granularity == "month" else "Y"
                    agg_df = detailed_df.groupby(pd.Grouper(key="date", freq=freq))["cost"].sum().reset_index()

                agg_df.rename(columns={"date": "ds", "cost": "y"}, inplace=True)
                agg_df['ds'] = pd.to_datetime(agg_df['ds'])
                agg_df = agg_df.sort_values('ds')

                # Log aggregation complete
                print(f"[{datetime.now()}] [{granularity}] Data aggregation complete. Data points: {len(agg_df)}")

                if len(agg_df) < SEQUENCE_LENGTH:
                    print(f"[{granularity}] Not enough data points ({len(agg_df)}) to train model. Need at least {SEQUENCE_LENGTH}. Skipping.")
                    continue

                # Create and train the model
                print(f"[{granularity}] Creating and training new Transformer Autoencoder...")
                model = TransformerAutoencoder(
                    seq_length=SEQUENCE_LENGTH,
                    num_layers=2,
                    d_model=64,
                    num_heads=8,
                    dff=128
                )
                # Log training start
                print(f"[{datetime.now()}] [{granularity}] Model training started...")
                model.train(agg_df) # This handles all preprocessing and training
                # Log training completion
                print(f"[{datetime.now()}] [{granularity}] Model training finished.")

                # Cache the trained model instance
                with cache_lock:
                    model_cache[granularity] = model
                print(f"[{granularity}] Model trained and cached successfully.")

        # Log any exception during the retraining cycle
        except Exception as e:
            print(f"[{datetime.now()}] An error occurred during the retraining cycle: {e}")

        print(f"[{datetime.now()}] Next model retraining in {timedelta(seconds=RETRAIN_INTERVAL_SECONDS)}.")
        time.sleep(RETRAIN_INTERVAL_SECONDS)

# --- API Routes ---
@app.route("/")
def hello_world():
    """Example Hello World route."""
    name = os.environ.get("NAME", "World")
    return f"Hello {name}!"

@app.route("/forecast", methods=["GET"])
def forecast():
    """Performs time series forecasting and anomaly detection."""
    granularity = request.args.get("granularity", "day").lower()
    if granularity not in GRANULARITIES:
        return jsonify({"error": "Invalid granularity. Choose 'day', 'month', or 'year'."}), 400

    with cache_lock:
        model = model_cache.get(granularity)

    if not model:
        return jsonify({"error": f"The model for '{granularity}' granularity is not ready yet. Please wait for training to complete."}), 503

    try:
        # Fetch fresh data for prediction
        detailed_data = fetch_detailed_cost_data()
        if detailed_data.empty:
            return jsonify({"forecast": [], "anomalies": [], "message": "No data available to make a forecast."})

        # Aggregate data to the required granularity
        if granularity == "day":
            agg_df = detailed_data.groupby(detailed_data["date"].dt.date)["cost"].sum().reset_index()
        else:
            freq = "M" if granularity == "month" else "Y"
            agg_df = detailed_data.groupby(pd.Grouper(key="date", freq=freq))["cost"].sum().reset_index()

        agg_df.rename(columns={"date": "ds", "cost": "y"}, inplace=True)
        agg_df['ds'] = pd.to_datetime(agg_df['ds'])
        agg_df = agg_df.sort_values('ds').reset_index(drop=True)

        # *** THE FIX: Call the actual prediction method on the model ***
        prediction_results = model.predict_anomalies(agg_df)

        return jsonify(prediction_results)

    except Exception as e:
        print(f"Error during prediction for '{granularity}': {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": "An error occurred during forecasting."}), 500

# ... (The rest of your endpoints: /analysis, /fetch_aws_cost, etc. remain unchanged)
# Make sure to include them here.

if __name__ == "__main__":
    retrain_thread = threading.Thread(target=retrain_models, daemon=True)
    retrain_thread.start()
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))