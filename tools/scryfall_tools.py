"""Scryfall download and refresh helpers for the CLI."""

import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry


SF_API_URL = "https://api.scryfall.com"
SF_API_HEADERS = {
    "User-Agent": "card-tagger/0.1",
    "Accept": "*/*",
}


def build_session() -> requests.Session:
    """Create a requests session with retries for Scryfall calls."""
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=10,
        pool_maxsize=50,
        max_retries=Retry(total=3, backoff_factor=0.3),
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def clean_cardname(name: str) -> str:
    """Create a filesystem-safe card name."""
    return (
        name.replace(" ", "_")
        .replace("/", "_")
        .replace(":", "_")
        .replace("'", "")
        .replace('"', "")
        .replace("?", "")
        .replace("!", "")
    )


def clear_folder(folder: Path, gitkeep_path: Path) -> None:
    """Remove folder contents while preserving the .gitkeep file."""
    if folder.exists():
        for item in folder.iterdir():
            if item != gitkeep_path:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
    else:
        folder.mkdir(parents=True, exist_ok=True)

    gitkeep_path.write_text("", encoding="utf-8")


def get_uris(session: requests.Session) -> dict:
    """Fetch bulk download URIs from Scryfall."""
    response = session.get(f"{SF_API_URL}/bulk-data", headers=SF_API_HEADERS)
    response.raise_for_status()
    data = response.json()["data"]

    download_uri = {}
    for item in data:
        download_uri[item["type"]] = item["download_uri"]
    return download_uri


def download_image(session: requests.Session, art_path: Path, entry: dict, is_dfc: bool = False, card_face: Optional[dict] = None) -> None:
    """Download a card image from Scryfall bulk data."""
    if not is_dfc:
        filename = f"{clean_cardname(entry['name'])}_{entry['illustration_id']}.jpg"
        image_url = entry["image_uris"]["art_crop"]
    else:
        filename = f"{clean_cardname(card_face['name'])}_{card_face['illustration_id']}.jpg"
        image_url = card_face["image_uris"]["art_crop"]

    file_path = art_path / filename
    response = session.get(image_url, headers=SF_API_HEADERS)
    if response.status_code != 200:
        logging.error("Failed to download image for %s: %s", entry["name"], response.status_code)
        return

    file_path.write_bytes(response.content)


def refresh_scryfall_data(data_path: Path) -> None:
    """Refresh Scryfall bulk data and download card art."""
    session = build_session()
    art_path = data_path / "art"
    data_gitkeep = data_path / ".gitkeep"
    art_gitkeep = art_path / ".gitkeep"

    clear_folder(data_path, data_gitkeep)
    logging.info("%s folder cleared", data_path)

    clear_folder(art_path, art_gitkeep)
    logging.info("%s folder cleared", art_path)

    artwork_file_path = data_path / "unique_artwork.json"
    download_uri = get_uris(session)

    response = session.get(download_uri["unique_artwork"], headers=SF_API_HEADERS)
    response.raise_for_status()
    artwork_file_path.write_bytes(response.content)

    with artwork_file_path.open("r", encoding="utf-8") as file:
        card_data = json.load(file)

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for entry in card_data:
            if "image_uris" in entry and "illustration_id" in entry:
                futures.append(executor.submit(download_image, session, art_path, entry))
            elif "card_faces" in entry:
                for card_face in entry["card_faces"]:
                    if "image_uris" in card_face and "illustration_id" in card_face:
                        futures.append(executor.submit(download_image, session, art_path, entry, True, card_face))
            else:
                logging.error("Unknown card type for %s", entry.get("name", "<unknown>"))

        logging.info("Downloads queued, waiting for completion...")
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading art", unit="img"):
            future.result()
