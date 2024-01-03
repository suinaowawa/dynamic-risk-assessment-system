from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json
import shutil


##################Load config.json and correct path variable
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
prod_deployment_path = os.path.join(config["prod_deployment_path"])

if not os.path.exists(prod_deployment_path):
    os.makedirs(prod_deployment_path)

model_path = os.path.join(config["output_model_path"], "trainedmodel.pkl")
latest_score_path = os.path.join(config["output_model_path"], "latestscore.txt")
ingested_files_path = os.path.join(config["output_folder_path"], "ingestedfiles.txt")


####################function for deployment
def store_model_into_pickle():
    """Copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory"""
    shutil.copy(model_path, os.path.join(prod_deployment_path, "trainedmodel.pkl"))
    shutil.copy(
        latest_score_path, os.path.join(prod_deployment_path, "latestscore.txt")
    )
    shutil.copy(
        ingested_files_path, os.path.join(prod_deployment_path, "ingestedfiles.txt")
    )
    print(f"Model, latest score, and ingested files copied to {prod_deployment_path}")


if __name__ == "__main__":
    store_model_into_pickle()
