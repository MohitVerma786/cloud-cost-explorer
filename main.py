import os
import threading
import time
from datetime import datetime, timedelta
import json

import pandas as pd
import psycopg2
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from prophet import Prophet
from psycopg2.pool import SimpleConnectionPool

import google.generativeai as genai
# Assuming fetch_aws_cost.py exists and contains the get_aws_cost_data function
# from fetch_aws_cost import get_aws_cost_data

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

# --- Gemini API Configuration ---
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

genai.configure(api_key=gemini_api_key)

# Create a single connection pool when the app starts
conn_pool = SimpleConnectionPool(
    minconn=1,
    maxconn=10,
    host=db_host,
    port=db_port,
    user=db_user,
    password=db_password,
    database=db_database,
)

# --- Model Caching & Background Training Infrastructure ---
cached_data = {}
cache_lock = threading.Lock()
GRANULARITIES = ["day", "month", "year"]
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

        df["date"] = pd.to_datetime(df["date"])
        df["cost"] = df["cost"].astype(float)
        return df
    finally:
        if conn:
            conn_pool.putconn(conn)


def fetch_detailed_cost_data_in_range(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches raw, unaggregated cost data from the database within a date range."""
    sql_query = """
        SELECT date, cost, service, region, usage_type, operation, linked_account
        FROM aws_cost_details
        WHERE date BETWEEN %s AND %s
        ORDER BY date;
    """
    conn = None
    try:
        conn = conn_pool.getconn()
        df = pd.read_sql_query(sql_query, conn, params=(start_date, end_date))
        if df.empty:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["date"])
        df["cost"] = df["cost"].astype(float)
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

                if granularity == "day":
                    agg_df = (
                        detailed_df.groupby(detailed_df["date"].dt.date)["cost"]
                        .sum()
                        .reset_index()
                    )
                else:
                    freq = "M" if granularity == "month" else "Y"
                    agg_df = (
                        detailed_df.groupby(pd.Grouper(key="date", freq=freq))["cost"]
                        .sum()
                        .reset_index()
                    )

                agg_df.rename(columns={"date": "ds", "cost": "y"}, inplace=True)

                if len(agg_df) < 2:
                    print(
                        f"[{granularity}] Not enough aggregated data points to train model. Skipping."
                    )
                    continue

                holidays_df = None
                if granularity == "day" and not agg_df.empty:
                    future_period = timedelta(days=365 * 5)
                    date_range = pd.date_range(
                        start=agg_df["ds"].min(), end=agg_df["ds"].max() + future_period
                    )
                    billing_days = date_range[date_range.day == 1]
                    holidays_df = pd.DataFrame(
                        {
                            "holiday": "monthly_billing",
                            "ds": pd.to_datetime(billing_days),
                            "lower_window": 0,
                            "upper_window": 0,
                        }
                    )

                print(f"[{granularity}] Fitting new model with custom holidays...")
                model = Prophet(holidays=holidays_df, interval_width=0.99)
                model.fit(agg_df)

                with cache_lock:
                    cached_data[granularity] = {
                        "model": model,
                        "details": detailed_df, # Store full details for all granularities
                    }
                print(f"[{granularity}] Model updated successfully.")
        except Exception as e:
            print(f"An error occurred during the retraining cycle: {e}")

        print(
            f"Next model retraining in {timedelta(seconds=RETRAIN_INTERVAL_SECONDS)}."
        )
        time.sleep(RETRAIN_INTERVAL_SECONDS)


# --- API Routes ---

@app.route("/")
def hello_world():
    """Example Hello World route."""
    name = os.environ.get("NAME", "World")
    return f"Hello {name}!"


@app.route("/forecast", methods=["GET"])
def forecast():
    """
    Performs time series forecasting and provides detailed anomaly detection.
    """
    granularity = request.args.get("granularity", "day").lower()
    if granularity not in GRANULARITIES:
        return (
            jsonify(
                {"error": "Invalid granularity. Choose 'day', 'month', or 'year'."}
            ),
            400,
        )

    with cache_lock:
        model_data = cached_data.get(granularity)

    if not model_data:
        return (
            jsonify(
                {
                    "error": f"The model or data for '{granularity}' granularity is not ready yet."
                }
            ),
            503,
        )

    model = model_data["model"]
    detailed_data = model_data["details"]

    try:
        periods, freq = (
            (12, "M")
            if granularity == "month"
            else (3, "Y") if granularity == "year" else (30, "D")
        )

        future = model.make_future_dataframe(periods=periods, freq=freq)
        full_forecast_df = model.predict(future)

        results_df = pd.merge(full_forecast_df, model.history, on="ds", how="left")

        anomalies_df = results_df[
            (results_df["y"] > results_df["yhat_upper"])
            | (results_df["y"] < results_df["yhat_lower"])
        ]

        anomalies_list = []
        if not anomalies_df.empty and not detailed_data.empty:
            if granularity == "day":
                anomalous_dates = anomalies_df["ds"].dt.date
                detailed_anomalies = detailed_data[
                    detailed_data["date"].dt.date.isin(anomalous_dates)
                ]
                meaningful_anomalies = detailed_anomalies[detailed_anomalies["cost"] > 0.0]
                anomalies_list = meaningful_anomalies.to_dict(orient="records")

            else:
                for _, row in anomalies_df.iterrows():
                    start_date = row["ds"]
                    end_date = (
                        (start_date + relativedelta(months=1) - timedelta(days=1))
                        if granularity == "month"
                        else (start_date + relativedelta(years=1) - timedelta(days=1))
                    )

                    period_details = detailed_data[
                        detailed_data["date"].between(start_date, end_date)
                    ]
                    meaningful_details = period_details[period_details["cost"] > 0.0]

                    anomalies_list.append(
                        {
                            "granularity": granularity,
                            "startDate": start_date.strftime("%Y-%m-%d"),
                            "endDate": end_date.strftime("%Y-%m-%d"),
                            "totalAnomalousCost": row["y"],
                            "costDetails": meaningful_details.to_dict(orient="records"),
                        }
                    )

        full_forecast_df.rename(
            columns={
                "ds": "date",
                "yhat": "predictedCost",
                "yhat_lower": "minExpectedCost",
                "yhat_upper": "maxExpectedCost",
            },
            inplace=True,
        )
        cost_columns = ["predictedCost", "minExpectedCost", "maxExpectedCost"]
        full_forecast_df[cost_columns] = full_forecast_df[cost_columns].clip(lower=0)
        future_forecast_df = full_forecast_df[
            full_forecast_df["date"].dt.date >= datetime.now().date()
        ]
        response_data = future_forecast_df[
            ["date", "predictedCost", "minExpectedCost", "maxExpectedCost"]
        ].to_dict(orient="records")

        # Standardize date formats in the response
        for record in response_data:
            record["date"] = record["date"].strftime("%Y-%m-%d")
        for anomaly in anomalies_list:
            if "date" in anomaly and isinstance(anomaly["date"], pd.Timestamp):
                anomaly["date"] = anomaly["date"].strftime("%Y-%m-%d %H:%M:%S")
            if "costDetails" in anomaly:
                for detail in anomaly["costDetails"]:
                     if "date" in detail and isinstance(detail["date"], pd.Timestamp):
                        detail["date"] = detail["date"].strftime("%Y-%m-%d %H:%M:%S")

        return jsonify({"forecast": response_data, "anomalies": anomalies_list})
    except Exception as e:
        print(f"Error during prediction for '{granularity}': {e}")
        return jsonify({"error": "An error occurred during forecasting."}), 500


@app.route("/analysis", methods=["GET"])
def analysis():
    """
    Provides a mock analysis endpoint mimicking the structure of generateMockAnalysisData.
    """
    granularity = request.args.get("granularity", "day").lower()
    startDate = request.args.get("startDate")
    endDate = request.args.get("endDate")

    if not startDate or not endDate:
        return jsonify({"error": "Please provide both startDate and endDate."}), 400

    try:
        start_date_obj = datetime.strptime(startDate, "%Y-%m-%d").date()
        end_date_obj = datetime.strptime(endDate, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Invalid date format. Please use YYYY-MM-DD."}), 400

    try:
        detailed_data_df = fetch_detailed_cost_data_in_range(startDate, endDate)

        if detailed_data_df.empty:
            return jsonify({
                "detailedCostData": [], "costData": [], "anomalies": [],
                "anomalyDetails": "No data available for analysis.",
                "recommendations": "No data available for recommendations."
            }), 200

        detailed_data_df['date_col'] = pd.to_datetime(detailed_data_df['date']).dt.date
        filtered_detailed_df = detailed_data_df[
            (detailed_data_df['date_col'] >= start_date_obj) &
            (detailed_data_df['date_col'] <= end_date_obj)
        ]

        detailed_data_list = filtered_detailed_df.drop(columns=['date_col']).to_dict(orient='records')
        for item in detailed_data_list:
            if isinstance(item.get('date'), pd.Timestamp):
                item['date'] = item['date'].strftime('%Y-%m-%d')

    except Exception as e:
        print(f"Error fetching data for analysis: {e}")
        return jsonify({"error": "Failed to retrieve data for analysis."}), 500

    forecast_response = forecast()
    if forecast_response.status_code != 200:
        return forecast_response

    anomalies_data = forecast_response.get_json().get('anomalies', [])

    aggregated_costs = {}
    for index, item in filtered_detailed_df.iterrows():
        item_date = item['date_col']
        if granularity == 'day':
            period = item_date.strftime("%Y-%m-%d")
        elif granularity == 'month':
            period = item_date.strftime("%Y-%m")
        elif granularity == 'year':
            period = item_date.strftime("%Y")
        else:
            period = item_date.strftime("%Y-%m-%d")

        if period not in aggregated_costs:
            aggregated_costs[period] = 0.0
        aggregated_costs[period] += item['cost']

    cost_data = []
    for period, cost in aggregated_costs.items():
        if granularity == 'month':
            date_obj = datetime.strptime(period, "%Y-%m")
        elif granularity == 'year':
            date_obj = datetime.strptime(period, "%Y")
        else:
            date_obj = datetime.strptime(period, "%Y-%m-%d")
        cost_data.append({"date": date_obj.strftime("%Y-%m-%d"), "cost": round(cost, 2), "service": "Total", "region": "N/A"})

    anomaly_details = "No anomalies detected in the selected range."
    recommendations = "No recommendations available as no anomalies were found."

    if anomalies_data:
        anomalous_cost_details = []
        for anomaly in anomalies_data:
            if 'costDetails' in anomaly:
                anomalous_cost_details.extend(anomaly['costDetails'])
            elif 'date' in anomaly and 'cost' in anomaly:
                 anomalous_cost_details.append(anomaly)

        if anomalous_cost_details:
            anomaly_summary = "Summary of detected cost anomalies:\n\n"
            for detail in anomalous_cost_details:
                anomaly_summary += f"- Date: {detail.get('date', 'N/A')}, Cost: {detail.get('cost', 'N/A'):.6f}, Service: {detail.get('service', 'N/A')}, Region: {detail.get('region', 'N/A')}, Usage Type: {detail.get('usage_type', 'N/A')}, Operation: {detail.get('operation', 'N/A')}\n"

            prompt = f"""
            You are an expert AWS cost analyst. Your task is to analyze a list of cost items that a statistical model has flagged as anomalous.

            Based on the data provided, perform two tasks:
            1.  Write a brief, one-paragraph summary explaining the potential root cause of this unexpected usage.
            2.  Provide a list of 2-3 actionable recommendations for an engineer to investigate these costs.

            **CRITICAL:** You MUST format your entire response as a single, valid JSON object with no other text before or after it.
            The JSON object must have exactly two keys: "anomalyDetails" and "recommendations".

            -   The value for "anomalyDetails" must be a string containing your summary.
            -   The value for "recommendations" must be an array of JSON objects.
            -   Each object in the "recommendations" array must have exactly two string keys: "title" (a short, actionable title) and "description" (a one or two-sentence explanation).

            **Anomaly Data:**
            {anomaly_summary}
            """
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                gemini_response = model.generate_content(prompt)
                
                try:
                    # Clean the response text before parsing
                    cleaned_text = gemini_response.text.strip().lstrip('```json').rstrip('```')
                    parsed_response = json.loads(cleaned_text)
                    anomaly_details = parsed_response.get("anomalyDetails", "Could not generate anomaly details.")
                    recommendations = parsed_response.get("recommendations", "Could not generate recommendations.")
                except json.JSONDecodeError:
                    print(f"Warning: Gemini response is not perfect JSON. Response text: {gemini_response.text}")
                    # Fallback if JSON parsing fails
                    anomaly_details = "Could not parse anomaly details from Gemini response."
                    recommendations = ["Could not parse recommendations from Gemini response."]

            except Exception as e:
                print(f"Error calling Gemini API: {e}")
                anomaly_details = "Could not generate anomaly details due to API error."
                recommendations = ["Could not generate recommendations due to API error."]

    final_response = {
        "detailedCostData": detailed_data_list,
        "costData": cost_data,
        "anomalies": anomalies_data,
        "anomalyDetails": anomaly_details,
        "recommendations": recommendations
    }
    return jsonify(final_response)


# Dummy function for get_aws_cost_data if the import fails or is not available
def get_aws_cost_data(start_date, end_date):
    print(f"Fetching dummy AWS cost data from {start_date} to {end_date}")
    return [] # Return empty list to simulate no data

@app.route("/fetch_aws_cost", methods=["GET"])
def fetch_aws_cost():
    """Fetches detailed AWS cost data, parses it, and stores it in the database."""
    start_date_str = request.args.get("start_date")
    end_date_str = request.args.get("end_date")

    if not start_date_str or not end_date_str:
        return jsonify({"error": "Please provide both start_date and end_date."}), 400

    try:
        datetime.strptime(start_date_str, "%Y-%m-%d")
        datetime.strptime(end_date_str, "%Y-%m-%d")
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
            host=db_host, port=db_port, user=db_user,
            password=db_password, database=db_database
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
                        dimensions_map.get("SERVICE", "N/A"),
                        dimensions_map.get("REGION", "N/A"),
                        dimensions_map.get("USAGE_TYPE", "N/A"),
                        dimensions_map.get("OPERATION", "N/A"),
                        dimensions_map.get("LINKED_ACCOUNT", "N/A"),
                        cost_amount,
                        unit,
                    ))
                except (ValueError, TypeError):
                    continue

        if not data_to_insert:
            return jsonify({"message": "No valid data to insert after processing."}), 200

        insert_query = """
            INSERT INTO aws_cost_details (date, service, region, usage_type, operation, linked_account, cost, unit)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (date, service, region, usage_type, operation, linked_account) DO UPDATE SET cost = EXCLUDED.cost;
        """
        psycopg2.extras.execute_batch(cur, insert_query, data_to_insert)
        conn.commit()

        return jsonify({"message": f"Successfully stored/updated {len(data_to_insert)} AWS cost records."}), 200

    except Exception as e:
        if conn:
            conn.rollback()
        print(f"Error storing data in database: {e}")
        return jsonify({"error": "An error occurred while storing data."}), 500
    finally:
        if conn:
            cur.close()
            conn.close()

@app.route("/chat", methods=["POST"])
def chat():
    """
    Accepts a question, gets forecast data, and uses Gemini to answer.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    # Get the forecast data by calling the forecast function
    # We need to mock a request context for the forecast function to work
    with app.test_request_context('/forecast?granularity=month'):
        forecast_response = forecast()
        forecast_data = forecast_response.get_json()


    prompt = f"""
    Based on the following forecast data, please answer the user's question.

    **Forecast Data:**
    ```json
    {json.dumps(forecast_data, indent=2)}
    ```

    **User's Question:**
    {question}
    """

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        gemini_response = model.generate_content(prompt)
        return jsonify({"answer": gemini_response.text})

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return jsonify({"error": "Could not generate an answer due to an API error."}), 500


if __name__ == "__main__":
    retrain_thread = threading.Thread(target=retrain_models, daemon=True)
    retrain_thread.start()
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))
