from pathlib import Path
import json

json_dir = Path("docs")
output_dir = Path("docs")
log_file = output_dir / "failure_log.txt"
log = []

count = 0

for file in json_dir.glob("*.json"):
    try:
        data = json.loads(file.read_text(encoding="utf-8"))

        # Fallbacks for missing fields
        title = data.get("title", {}).get("@value") or f"Untitled ICD Entry [{file.stem}]"
        synonyms = [
            s.get("label", {}).get("@value", "")
            for s in data.get("synonym", [])
            if s.get("label", {}).get("@value")
        ]
        definition = data.get("definition", {}).get("@value")
        browser_url = data.get("browserUrl", "")

        # Build the full text
        parts = [f" Title: {title}"]
        if synonyms:
            parts.append(" Synonyms:\n- " + "\n- ".join(synonyms))
        if definition:
            parts.append(f" Definition:\n{definition}")
        else:
            parts.append(" No definition provided for this ICD entity.")

        if browser_url:
            parts.append(f" Source: {browser_url}")

        full_text = "\n\n".join(parts)

        # Sanitize filename
        filename = "".join(c for c in title if c.isalnum() or c in " _-")[:60]
        (output_dir / f"{filename}.txt").write_text(full_text, encoding="utf-8")
        count += 1

        if not definition:
            log.append(f"{file.name} → missing_definition")

    except Exception as e:
        log.append(f"{file.name} → exception: {str(e)}")

# Save log
log_file.write_text("\n".join(log), encoding="utf-8")

print(f" Converted {count} ICD entries into .txt")
print(f" Logged {len(log)} issues → see {log_file}")
