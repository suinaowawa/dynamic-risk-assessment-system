import os
import requests
import json

# Specify the URL for your Flask app
URL = "http://127.0.0.1:8000"

# API endpoints
prediction_endpoint = f"{URL}/prediction"
scoring_endpoint = f"{URL}/scoring"
summarystats_endpoint = f"{URL}/summarystats"
diagnostics_endpoint = f"{URL}/diagnostics"

# API calls
response1 = requests.post(
    prediction_endpoint, json={"file_location": "testdata/testdata.csv"}
)
response2 = requests.get(scoring_endpoint)
response3 = requests.get(summarystats_endpoint)
response4 = requests.get(diagnostics_endpoint)


# Combine responses
combined_responses = {
    "prediction": response1.json(),
    "scoring": response2.json(),
    "summarystats": response3.json(),
    "diagnostics": response4.json(),
}

# Write combined responses to apireturns.txt
with open("config.json", "r") as f:
    config = json.load(f)

output_path = os.path.join(config["output_model_path"], "apireturns.txt")
with open(output_path, "w") as output_file:
    json.dump(combined_responses, output_file, indent=2)

print(f"Combined responses written to {output_path}")
