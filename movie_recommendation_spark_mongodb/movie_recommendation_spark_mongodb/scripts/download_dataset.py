from pathlib import Path
import urllib.request
import zipfile
import io

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
TARGET = DATA_ROOT / "ml-100k"
URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

def main():
    if (TARGET / "u.data").exists() and (TARGET / "u.item").exists():
        print(f"Dataset already exists: {TARGET}")
        return
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    print("Downloading MovieLens 100K from GroupLens...")
    request = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        archive_bytes = response.read()
    print("Extracting dataset...")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zf:
        zf.extractall(DATA_ROOT)
    if not (TARGET / "u.data").exists():
        raise FileNotFoundError(f"Expected u.data inside {TARGET}")
    print(f"Dataset ready at: {TARGET}")

if __name__ == "__main__":
    main()
