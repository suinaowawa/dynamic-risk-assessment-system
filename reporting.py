import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from diagnostics import model_predictions


###############Load config.json and get path variables
with open("config.json", "r") as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config["output_folder_path"])
test_data_path = os.path.join(config["test_data_path"], "testdata.csv")
output_plot_path = os.path.join(config["output_model_path"], "confusionmatrix.png")


##############Function for reporting
def score_model():
    """
    calculate a confusion matrix using the test data and the deployed model
    write the confusion matrix to the workspace
    """
    input_df = pd.read_csv(test_data_path)
    predictions = model_predictions(input_df)
    actual_values = input_df["exited"].values

    cm = confusion_matrix(actual_values, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    # Save confusion matrix plot to a file
    plt.savefig(output_plot_path, format="png")
    print(f"Confusion matrix plot saved to {output_plot_path}")
    plt.show()


if __name__ == "__main__":
    score_model()
