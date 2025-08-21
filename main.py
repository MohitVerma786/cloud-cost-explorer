import os

import json
from flask import Flask, request, jsonify
import pandas as pd
from prophet import Prophet

app = Flask(__name__)

@app.route("/")
def hello_world():
  """Example Hello World route."""
  name = os.environ.get("NAME", "World")
  return f"Hello {name}!"

@app.route("/forecast", methods=['POST'])
def forecast():
 """Performs time series forecasting using Prophet."""
 if not request.json or 'ds' not in request.json or 'y' not in request.json:
  return jsonify({"error": "Invalid request body. 'ds' and 'y' data required."}), 400

 data = {
 'ds': request.json['ds'],
 'y': request.json['y']
  }
 df = pd.DataFrame(data)
 model = Prophet()
 model.fit(df)
 future = model.make_future_dataframe(periods=30)
 forecast = model.predict(future)
 return jsonify(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records'))

if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 3000)))