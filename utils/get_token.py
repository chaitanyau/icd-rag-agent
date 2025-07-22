import requests

# WHO ICD Token Endpoint
TOKEN_URL = "https://icdaccessmanagement.who.int/connect/token"


CLIENT_ID = ""
CLIENT_SECRET = ""


response = requests.post(TOKEN_URL, data={
    "client_id": CLIENT_ID,
    "client_secret": CLIENT_SECRET,
    "grant_type": "client_credentials",
    "scope": "icdapi_access"
})

# Debug: Check full response
print("Status Code:", response.status_code)
print("Raw Response:", response.text)

# Extract token only if successful
if response.status_code == 200:
    token = response.json()["access_token"]
    print("\n✅ Bearer Token:")
    print(token)
else:
    print("\n Failed to retrieve access token. Please check credentials or network.")
