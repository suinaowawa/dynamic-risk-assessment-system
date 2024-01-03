import subprocess
from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle

from diagnostics import (
    check_missing_data,
    dataframe_summary,
    execution_time,
    model_predictions,
    outdated_packages_list,
)

import json
import os


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = "1652d576-484a-49fd-913a-6879acfa6ba4"

with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=["POST", "OPTIONS"])
def predict():
    # call the prediction function you created in Step 3
    try:
        # Get the file location from the request
        file_location = request.json.get("file_location")

        if not file_location:
            return jsonify({"error": "Missing 'file_location' parameter"}), 400

        # Load the dataset from the specified file location
        dataset = pd.read_csv(file_location)

        # Call the prediction function
        predictions = model_predictions(dataset)

        # Return the prediction outputs as JSON
        return jsonify({"predictions": predictions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


#######################Scoring Endpoint
@app.route("/scoring", methods=["GET", "OPTIONS"])
def scoring():
    # check the score of the deployed model
    try:
        # Run the scoring.py script using subprocess
        result = subprocess.run(
            ["python", "scoring.py"], capture_output=True, text=True, check=True
        )

        # Extract the F1 score from the script's output
        f1_score = result.stdout.strip()

        # Return the F1 score as JSON
        return jsonify({"f1_score": f1_score})

    except subprocess.CalledProcessError as e:
        return jsonify({"error": f"Error running scoring.py: {e.stderr}"}), 500


#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=["GET", "OPTIONS"])
def stats():
    # check means, medians, and modes for each column
    try:
        # Call the summary statistics function
        summary_stats = dataframe_summary()

        # Return the summary statistics as JSON
        return jsonify(summary_stats)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=["GET", "OPTIONS"])
def diagnostics():
    # check timing and percent NA values and outdated dependencies
    try:
        # Measure execution time
        timings = execution_time()

        # Check missing data
        missing_data = check_missing_data()

        # Check dependencies
        outdated_dependencies = outdated_packages_list().to_dict(orient="records")

        # Combine results into a single dictionary
        diagnostics_data = {
            "execution_time": timings,
            "missing_data_percentage": missing_data,
            "outdated_dependencies": outdated_dependencies,
        }

        # Return the diagnostics data as JSON
        return jsonify(diagnostics_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True, threaded=True)
