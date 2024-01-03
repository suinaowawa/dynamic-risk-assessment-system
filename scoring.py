from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json


#################Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"], "testdata.csv")
model_path = os.path.join(config["output_model_path"], "trainedmodel.pkl")
latest_score_path = os.path.join(config["output_model_path"], "latestscore.txt")


#################Function for model scoring
def score_model():
    """
    this function should take a trained model, load test data,
    and calculate an F1 score for the model relative to the test data
    it should write the result to the latestscore.txt file
    """
    # Load the test data
    test_df = pd.read_csv(test_data_path)

    # Load the trained model
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)

    # Define features and target variable
    features = test_df[
        ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    ]
    target = test_df["exited"]

    # Make predictions on the test set
    y_pred = model.predict(features)

    # Calculate the F1 score
    f1_score = metrics.f1_score(target, y_pred)

    # Write the F1 score to latestscore.txt
    with open(latest_score_path, "w") as score_file:
        score_file.write(str(f1_score))

    print(f"F1 Score: {f1_score} - Latest score saved to {latest_score_path}")


if __name__ == "__main__":
    score_model()
