import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


#############Load config.json and get input and output paths
with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
output_folder_path = config["output_folder_path"]

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)


#############Function for data ingestion
def merge_multiple_dataframe():
    """Check for datasets, compile them together, and write to an output file."""

    # Create an empty DataFrame to store the merged data
    merged_df = pd.DataFrame()

    # List to store the names of ingested CSV files
    ingested_files = []

    # Iterate through every CSV file in the input folder
    for file in os.listdir(input_folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(input_folder_path, file)
            df = pd.read_csv(file_path)

            # Check if the DataFrame has the same column names as the existing merged DataFrame
            if merged_df.empty or set(df.columns) == set(merged_df.columns):
                merged_df = pd.concat([merged_df, df], ignore_index=True)
                ingested_files.append(file)
            else:
                print(f"Skipping {file} due to mismatched column names.")

    # Remove duplicate rows from the merged DataFrame
    merged_df.drop_duplicates(inplace=True)

    # Save the final DataFrame to a CSV file in the output folder
    final_output_path = os.path.join(output_folder_path, "finaldata.csv")
    merged_df.to_csv(final_output_path, index=False)

    # Save the list of ingested files to ingestedfiles.txt
    ingested_files_path = os.path.join(output_folder_path, "ingestedfiles.txt")
    with open(ingested_files_path, "w") as ingested_file:
        ingested_file.write("\n".join(ingested_files))

    print("Merged and deduplicated data saved to finaldata.csv.")
    print("List of ingested files saved to ingestedfiles.txt.")


if __name__ == "__main__":
    merge_multiple_dataframe()
