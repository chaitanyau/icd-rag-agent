from pathlib import Path

json_dir = Path("docs/raw_json")
files = list(json_dir.glob("*.json"))
print(f"Found {len(files)} JSON files")
for f in files[:10]:
    print(f)