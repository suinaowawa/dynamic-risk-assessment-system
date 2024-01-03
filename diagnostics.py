import pickle
import subprocess
import pandas as pd
import numpy as np
import timeit
import os
import json

##################Load config.json and get environment variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"], "finaldata.csv")
test_data_path = os.path.join(config["test_data_path"], "testdata.csv")
prod_deployment_path = os.path.join(config["prod_deployment_path"])
model_path = os.path.join(prod_deployment_path, "trainedmodel.pkl")


##################Function to get model predictions
def model_predictions(input_df: pd.DataFrame) -> list:
    """read the deployed model and a test dataset, calculate predictions"""
    # Load the deployed model
    with open(model_path, "rb") as model_file:
        deployed_model = pickle.load(model_file)

    # Ensure the input_data has the same features as used during training
    features = input_df[
        ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    ]

    # Make predictions
    predictions = deployed_model.predict(features).tolist()

    return predictions


##################Function to get summary statistics
def dataframe_summary() -> dict:
    """
    Calculate summary statistics for a dataset.

    Returns:
    dict: A dictionary containing selected summary statistics (mean, median, standard deviation).
    """
    # calculate summary statistics here
    df = pd.read_csv(dataset_csv_path)
    summary_statistics = df.describe()
    selected_statistics = summary_statistics.loc[["mean", "50%", "std"]].to_dict(
        orient="index"
    )

    return selected_statistics


#################### Function for checking missing data
def check_missing_data() -> list:
    """
    Check for missing data in each column of the dataset.

    Returns:
    list: A list containing the percentage of missing values for each column.
    """
    # Load the dataset
    df = pd.read_csv(dataset_csv_path)

    # Calculate the number of missing values in each column
    missing_counts = df.isnull().sum()

    # Calculate the percentage of missing values in each column
    total_rows = len(df)
    missing_percentages = (missing_counts / total_rows) * 100

    # Convert the result to a list
    result_list = missing_percentages.tolist()

    return result_list


##################Function to get timings
def execution_time() -> list:
    """
    Time the data ingestion and model training tasks.

    Returns:
    list: A list containing two timing measurements in seconds (data ingestion time, model training time).
    """
    # Calculate timing of data ingestion
    ingestion_duration = timeit.timeit(
        lambda: subprocess.run(["python", "ingestion.py"], check=True), number=1
    )

    # Calculate timing of model training
    training_duration = timeit.timeit(
        lambda: subprocess.run(["python", "training.py"], check=True), number=1
    )

    return [ingestion_duration, training_duration]


##################Function to check dependencies
def outdated_packages_list() -> pd.DataFrame:
    """
    Check the currently installed and latest versions of outdated third-party Python modules.

    Returns:
    pd.DataFrame: A DataFrame containing information about outdated packages, including
                  the package name, installed version, and the latest available version.
    """
    try:
        result = subprocess.run(
            ["pip", "list", "--outdated"], stdout=subprocess.PIPE, text=True
        )
        outdated_packages = [
            line.split()[:3] for line in result.stdout.strip().split("\n")[2:]
        ]

        df = pd.DataFrame(
            outdated_packages,
            columns=["Package", "Installed Version", "Latest Version"],
        )
        return df

    except Exception as e:
        print(f"Error checking dependencies: {e}")


if __name__ == "__main__":
    input_df = pd.read_csv(test_data_path)
    print(model_predictions(input_df))
    print(dataframe_summary())
    print(check_missing_data())
    print(execution_time())
    print(outdated_packages_list())
