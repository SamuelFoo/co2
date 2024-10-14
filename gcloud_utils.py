import re
from pathlib import Path

import requests
from google.auth import default
from google.auth.transport.requests import Request

PI3_BUCKET_NAME = "co2-measurements-pi-3"
PI4_BUCKET_NAME = "co2-measurements-pi-4"


# Function to get an access token
def get_access_token():
    # Get default credentials
    credentials, _ = default()
    # Refresh the token if necessary
    credentials.refresh(Request())
    return credentials.token


def fetch_gcs_object(url):
    access_token = get_access_token()
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response
    else:
        print(f"Error fetching object: {response.status_code} - {response.text}")
        return None


def get_data_file_metadata(bucket):
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o"
    return fetch_gcs_object(url).json()


def get_all_data_files(save_dir: Path, bucket):
    save_dir.mkdir(parents=True, exist_ok=True)

    pi3_metadata = get_data_file_metadata(bucket)
    file_urls = list(map(lambda x: x["mediaLink"], pi3_metadata["items"]))

    for file_url in file_urls:
        response = fetch_gcs_object(file_url)

        pattern = r"\/([^\/]+%20[^\/]+\.csv)"
        match = re.search(pattern, file_url)
        file_name = match.group(1).replace(":", "_").replace("%20", " ")

        with open(save_dir / file_name, "w") as f:
            f.write(response.text.replace("\n", ""))
