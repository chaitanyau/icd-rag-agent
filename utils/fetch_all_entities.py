import requests
import os
import json
import time
from tqdm import tqdm

# ------------------- CONFIG -----------------------
#TOKEN = ""  # truncated for safety
HEADERS = {
    "Authorization": TOKEN,
    "Accept": "application/json",
    "API-Version": "v2",
    "Accept-Language": "en"
}
BASE_URL = "https://id.who.int/icd"
RELEASE_ID = "2025-01"
OUTPUT_DIR = "docs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
# --------------------------------------------------

fetched = set()


def entity_already_saved(uri):
    entity_id = uri.split("/")[-1]
    return os.path.exists(os.path.join(OUTPUT_DIR, f"{entity_id}.json"))


def fetch_entity(uri):
    if uri in fetched or entity_already_saved(uri):
        return None

    full_url = f"{uri}?releaseId={RELEASE_ID}"
    print(f"→ Fetching: {uri}")
    try:
        response = requests.get(full_url, headers=HEADERS)
        if response.status_code == 200:
            entity = response.json()
            entity_id = uri.split("/")[-1]
            with open(f"{OUTPUT_DIR}/{entity_id}.json", "w", encoding="utf-8") as f:
                json.dump(entity, f, indent=2, ensure_ascii=False)
            fetched.add(uri)
            return entity
        else:
            print(f"[ERROR] Failed: {uri} → {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"[EXCEPTION] {uri}: {e}")
        return None


def fetch_children_recursive(uri):
    entity = fetch_entity(uri)
    if not entity:
        return
    children = entity.get("child", [])
    for child_uri in tqdm(children, desc=f"↳ Children of {uri.split('/')[-1]}"):
        fetch_children_recursive(child_uri)
        time.sleep(0.1)  # polite delay


if __name__ == "__main__":
    
    top_jsons = [
        "docs/448895267.json",  
        "docs/1405434703.json"  
        # Add more here if needed
    ]

    for json_file in top_jsons:
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
            children = data.get("child", [])
            for child_uri in tqdm(children, desc=f"🌐 Root {json_file}"):
                fetch_children_recursive(child_uri)
