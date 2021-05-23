import pandas as pd
import requests
import time

ENDPOINT = "http://127.0.0.1:8000/predict"
REQUEST_FILE = "requests.csv"
NUM_REQUESTS = 100

if __name__ == "__main__":
    data = pd.read_csv(REQUEST_FILE)
    for i in range(NUM_REQUESTS):
        request_data = data.iloc[i].to_dict()
        request_data["id"] = i
        response = requests.get(
            ENDPOINT,
            json=[request_data],
        )
        print(f'Request: {request_data}')
        print(f'Response CODE: {response.status_code}')
        print(f'Response BODY: {response.json()}')
        