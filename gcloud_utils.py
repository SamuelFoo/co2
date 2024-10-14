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


def get_data_file(file_url):
    data = fetch_gcs_object(file_url)
