from pathlib import Path
import csv
import io
import time
import requests
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = BASE_DIR / "datasets" / "realworld_finetune" / "REAL"
METADATA_FILE = BASE_DIR / "datasets" / "realworld_finetune" / "metadata.csv"

TARGET_IMAGES = 2000

# Wikimedia Commons API
API_URL = "https://commons.wikimedia.org/w/api.php"

# Search terms give us diverse genuine photographs
SEARCH_TERMS = [
    "photograph people",
    "photograph animals",
    "photograph nature",
    "photograph landscape",
    "photograph buildings",
    "photograph vehicles",
    "photograph food",
    "photograph objects",
    "photograph sports",
    "photograph flowers",
    "photograph city",
    "photograph indoor",
    "photograph outdoor",
    "photograph travel",
    "photograph wildlife",
]

HEADERS = {
    "User-Agent": "DigitalMediaAuthenticityAssessment/1.0"
}

MAX_SIZE = 1024

# ============================================================
# CREATE DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("REAL-WORLD IMAGE DOWNLOADER")
print("=" * 60)
print(f"Target images : {TARGET_IMAGES}")
print(f"Output folder : {OUTPUT_DIR}")
print()

# ============================================================
# EXISTING FILES
# ============================================================

existing_files = list(OUTPUT_DIR.glob("*.jpg"))

print(f"Existing valid JPG files: {len(existing_files)}")

if len(existing_files) >= TARGET_IMAGES:
    print()
    print("Already have enough images.")
    print("Nothing more to download.")
    exit(0)

downloaded = len(existing_files)
failed = 0
seen_urls = set()

# ============================================================
# METADATA
# ============================================================

metadata_rows = []

if METADATA_FILE.exists():
    try:
        with open(METADATA_FILE, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            metadata_rows = list(reader)

        for row in metadata_rows:
            if row.get("image_url"):
                seen_urls.add(row["image_url"])

    except Exception:
        metadata_rows = []

# ============================================================
# SEARCH FUNCTION
# ============================================================

def search_images(search_term, continue_token=None):
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": search_term,
        "gsrnamespace": 6,
        "gsrlimit": 50,
        "prop": "imageinfo",
        "iiprop": "url|mime|size",
        "iiurlwidth": 1024,
        "format": "json",
    }

    if continue_token:
        params["gsrcontinue"] = continue_token

    response = requests.get(
        API_URL,
        params=params,
        headers=HEADERS,
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    pages = data.get("query", {}).get("pages", {})

    next_token = (
        data.get("continue", {})
        .get("gsrcontinue")
    )

    return pages, next_token


# ============================================================
# DOWNLOAD + VALIDATE
# ============================================================

def download_image(url):
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=30,
        )

        if response.status_code != 200:
            return None

        content_type = response.headers.get(
            "Content-Type", ""
        ).lower()

        if not content_type.startswith("image/"):
            return None

        image = Image.open(io.BytesIO(response.content))

        # Fully verify image
        image.verify()

        # Re-open after verify
        image = Image.open(io.BytesIO(response.content))
        image = image.convert("RGB")

        # Keep images reasonably sized
        image.thumbnail(
            (MAX_SIZE, MAX_SIZE),
            Image.Resampling.LANCZOS
        )

        return image

    except Exception:
        return None


# ============================================================
# MAIN DOWNLOAD LOOP
# ============================================================

for search_term in SEARCH_TERMS:

    if downloaded >= TARGET_IMAGES:
        break

    print()
    print("-" * 60)
    print(f"Searching: {search_term}")
    print("-" * 60)

    continue_token = None
    pages_checked = 0

    while downloaded < TARGET_IMAGES:

        try:
            pages, continue_token = search_images(
                search_term,
                continue_token
            )

        except Exception as e:
            print(f"Search error: {e}")
            break

        pages_checked += 1

        if not pages:
            break

        for page in pages.values():

            if downloaded >= TARGET_IMAGES:
                break

            imageinfo = page.get("imageinfo", [])

            if not imageinfo:
                continue

            info = imageinfo[0]

            image_url = info.get("thumburl") or info.get("url")
            mime = info.get("mime", "")

            if not image_url:
                continue

            if image_url in seen_urls:
                continue

            # Only common photographic formats
            if mime not in {
                "image/jpeg",
                "image/png",
                "image/webp",
            }:
                continue

            seen_urls.add(image_url)

            image = download_image(image_url)

            if image is None:
                failed += 1
                continue

            downloaded += 1

            filename = OUTPUT_DIR / f"real_{downloaded:04d}.jpg"

            try:
                image.save(
                    filename,
                    format="JPEG",
                    quality=92,
                    optimize=True
                )

                metadata_rows.append({
                    "filename": filename.name,
                    "source": "Wikimedia Commons",
                    "image_url": image_url,
                    "title": page.get("title", ""),
                    "search_term": search_term,
                })

            except Exception:
                downloaded -= 1
                failed += 1
                continue

            if downloaded % 25 == 0 or downloaded <= 10:
                print(
                    f"Downloaded: {downloaded}/{TARGET_IMAGES}"
                )

            # Be polite to Wikimedia's API
            time.sleep(0.15)

        if not continue_token:
            break

    print(
        f"Progress: {downloaded}/{TARGET_IMAGES}"
    )


# ============================================================
# SAVE METADATA
# ============================================================

with open(
    METADATA_FILE,
    "w",
    encoding="utf-8",
    newline=""
) as f:

    fieldnames = [
        "filename",
        "source",
        "image_url",
        "title",
        "search_term",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(metadata_rows)


# ============================================================
# FINAL CHECK
# ============================================================

final_files = list(OUTPUT_DIR.glob("*.jpg"))

print()
print("=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)
print(f"Valid REAL images : {len(final_files)}")
print(f"Failed downloads  : {failed}")
print(f"Images location   : {OUTPUT_DIR}")
print(f"Metadata          : {METADATA_FILE}")
print("=" * 60)

if len(final_files) < TARGET_IMAGES:
    print()
    print(
        f"WARNING: Only {len(final_files)} images were obtained."
    )
    print(
        "Run the same command again to continue downloading."
    )
else:
    print()
    print("SUCCESS: 2,000 REAL images are ready.")