import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_APTOS = PROJECT_ROOT / "data" / "raw" / "aptos"
RAW_MESS = PROJECT_ROOT / "data" / "raw" / "messidor"

OUT_APTOS = PROJECT_ROOT / "data" / "cache_224" / "aptos"
OUT_MESS = PROJECT_ROOT / "data" / "cache_224" / "messidor"

OUT_APTOS.mkdir(parents=True, exist_ok=True)
OUT_MESS.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
JPEG_QUALITY = 85

def center_square_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    m = min(w, h)
    left = (w - m) // 2
    top = (h - m) // 2
    return img.crop((left, top, left + m, top + m))

def save_224(src_path: Path, dst_path: Path):
    try:
        img = Image.open(src_path).convert("RGB")
        img = center_square_crop(img)
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path, format="JPEG", quality=JPEG_QUALITY, optimize=True)
        return True
    except Exception:
        return False

def find_aptos_images():
    # Your APTOS binary folder structure from Kaggle:
    # .../APTOS 2019 (Original) (Binary)/DR/*.png
    # .../APTOS 2019 (Original) (Binary)/No DR/*.png
    candidates = list(RAW_APTOS.rglob("*.png")) + list(RAW_APTOS.rglob("*.jpg")) + list(RAW_APTOS.rglob("*.jpeg"))
    return candidates

def find_messidor_images():
    # Messidor commonly has: .../messidor2/IMAGES/*.jpg or *.png
    candidates = list(RAW_MESS.rglob("*.jpg")) + list(RAW_MESS.rglob("*.png")) + list(RAW_MESS.rglob("*.jpeg"))
    return candidates

def main():
    print("Project:", PROJECT_ROOT)
    print("Caching to:")
    print("  APTOS  ->", OUT_APTOS)
    print("  MESSIDOR ->", OUT_MESS)

    # --- APTOS ---
    aptos_imgs = find_aptos_images()
    print(f"Found APTOS images: {len(aptos_imgs)}")
    ok = 0
    fail = 0

    for p in tqdm(aptos_imgs, desc="APTOS cache"):
        # add class folder to avoid collisions: DR_ or NoDR_
        parent = p.parent.name.replace(" ", "").replace("-", "")
        dst_name = f"{parent}_{p.stem}.jpg"
        dst = OUT_APTOS / dst_name

        if not dst.exists():
            if save_224(p, dst):
                ok += 1
            else:
                fail += 1

    print(f"APTOS cached new: {ok} | failed: {fail}")

    # --- MESSIDOR ---
    mess_imgs = find_messidor_images()
    print(f"Found MESSIDOR images: {len(mess_imgs)}")
    ok = 0
    for p in tqdm(mess_imgs, desc="MESSIDOR cache"):
        dst = OUT_MESS / (p.stem + ".jpg")
        if not dst.exists():
            ok += 1 if save_224(p, dst) else 0
    print(f"MESSIDOR cached new: {ok}")

    print("\n✅ Done.")
    print("Cache sizes:")
    os.system(f"du -sh {PROJECT_ROOT/'data/cache_224/aptos'} {PROJECT_ROOT/'data/cache_224/messidor'}")

if __name__ == "__main__":
    main()
