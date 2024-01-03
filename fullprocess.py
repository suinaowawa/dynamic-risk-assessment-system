#!/bin/bash cd ~/learn/mlops-udacity/project-3/dynamic-risk-assessment-system
import json
import os
import subprocess

import pandas as pd
from sklearn import metrics

from diagnostics import model_predictions


with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]
prod_deployment_path = config["prod_deployment_path"]


##################Check and read new data
# first, read ingestedfiles.txt
def check_new_data() -> bool:
    ingested_files_path = os.path.join(prod_deployment_path, "ingestedfiles.txt")
    try:
        with open(ingested_files_path, "r") as ingested_file:
            ingested_files = set(ingested_file.read().splitlines())
    except FileNotFoundError:
        ingested_files = set()

    # second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
    input_files = set(os.listdir(input_folder_path))

    if not input_files.issubset(ingested_files):
        subprocess.run(["python", "ingestion.py"])
        return True
    return False


##################Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here


##################Checking for model drift
# check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
def check_model_drift() -> bool:
    # Read the latest score from latestscore.txt
    latest_score_path = os.path.join(prod_deployment_path, "latestscore.txt")

    try:
        with open(latest_score_path, "r") as score_file:
            latest_score = float(score_file.read().strip())
    except FileNotFoundError:
        latest_score = 1.0

    ingested_data_path = os.path.join(output_folder_path, "finaldata.csv")
    input_df = pd.read_csv(ingested_data_path)
    predictions = model_predictions(input_df)

    target = input_df["exited"]

    # Calculate the new F1 score
    new_score = metrics.f1_score(target, predictions)
    print(f"New Score: {new_score}, Old Score: {latest_score}")
    # Check for model drift
    if new_score < latest_score:
        print("Model drift has occurred.")
        return True
    else:
        print("No model drift.")
        return False


##################Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the process here


##################Re-deployment
# if you found evidence for model drift, re-run the deployment.py script
def retrain_model():
    subprocess.run(["python", "training.py"])


def redeploy_model():
    subprocess.run(["python", "deployment.py"])


##################Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model
def run_diagnostics():
    subprocess.run(["python", "apicalls.py"])


def run_reporting():
    subprocess.run(["python", "reporting.py"])


def full_process():
    print("====checking new data")
    is_newdata = check_new_data()
    print("====is_newdata:", is_newdata)
    if not is_newdata:
        return
    print("====checking model drift")
    is_modeldrift = check_model_drift()
    print("====is_modeldrift:", is_modeldrift)
    if not is_modeldrift:
        return
    retrain_model()
    redeploy_model()
    run_diagnostics()
    run_reporting()
    return


if __name__ == "__main__":
    full_process()
