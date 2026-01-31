from pathlib import Path
from PIL import Image
import random


ROOT = Path(__file__).resolve().parents[1]
APTOS = ROOT / "data/cache_224/aptos"
MESS  = ROOT / "data/cache_224/messidor"

def sample_stats(folder: Path, n=10):
    files = sorted(folder.glob("*.jpg"))
    print(f"\nFolder: {folder}")
    print("Count:", len(files))
    if not files:
        return

    picks = random.sample(files, min(n, len(files)))

    sizes = []
    for p in picks:
        size_kb = p.stat().st_size / 1024
        sizes.append(size_kb)
        with Image.open(p) as im:
            w, h = im.size
        print(f"{p.name:35s}  {size_kb:8.1f} KB   {w}x{h}")

    print("\nAvg KB (sample):", sum(sizes)/len(sizes))
    print("Min KB (sample):", min(sizes))
    print("Max KB (sample):", max(sizes))

def folder_size(folder: Path):
    total = 0
    for p in folder.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024*1024)

if __name__ == "__main__":
    print("APTOS total size (MB):", round(folder_size(APTOS), 2))
    print("MESS total size (MB):", round(folder_size(MESS), 2))
    sample_stats(APTOS, n=12)
    sample_stats(MESS, n=12)
