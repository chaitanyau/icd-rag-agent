from pathlib import Path
print(len(list(Path("docs").glob("*.txt"))))
