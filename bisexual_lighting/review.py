"""
Open tagger.scryfall.com for every card labelled "Bisexual Lighting" or "yes"
in a Label Studio export, one at a time.

Refreshes data/unique_artwork.json from the Scryfall bulk-data API before
iterating so the set/collector-number lookup is always current.

Usage:
  python -m bisexual_lighting.review export.json
"""

from __future__ import annotations

import json
import logging
import re
import sys
import webbrowser
from pathlib import Path

from tools.scryfall_tools import build_session, get_uris, SF_API_HEADERS

log = logging.getLogger(__name__)

DATA_PATH = Path(__file__).resolve().parent.parent / "data"
ARTWORK_PATH = DATA_PATH / "unique_artwork.json"
TARGET_LABELS = {"bisexual lighting", "yes"}


def refresh_unique_artwork() -> None:
    log.info("Refreshing unique_artwork.json from Scryfall…")
    session = build_session()
    uris = get_uris(session)
    response = session.get(uris["unique_artwork"], headers=SF_API_HEADERS)
    response.raise_for_status()
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    ARTWORK_PATH.write_bytes(response.content)
    log.info("Saved %s (%.1f MB)", ARTWORK_PATH, ARTWORK_PATH.stat().st_size / 1_000_000)


def load_illustration_lookup() -> dict:
    with ARTWORK_PATH.open("r", encoding="utf-8") as f:
        cards = json.load(f)
    return {
        card["illustration_id"]: card
        for card in cards
        if "illustration_id" in card
    }


def extract_uuid(image_path: str) -> str | None:
    m = re.search(
        r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.jpg$",
        image_path,
    )
    return m.group(1) if m else None


def get_labelled_entries(export_path: Path) -> list[dict]:
    tasks = json.loads(export_path.read_text(encoding="utf-8"))
    results = []
    for task in tasks:
        for ann in task.get("annotations", []):
            if ann.get("was_cancelled"):
                continue
            for result in ann.get("result", []):
                choices = result.get("value", {}).get("choices", [])
                if any(c.lower() in TARGET_LABELS for c in choices):
                    results.append(task)
                    break
    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if len(sys.argv) < 2:
        log.error("Usage: python -m bisexual_lighting.review <export.json>")
        sys.exit(1)

    export_path = Path(sys.argv[1])
    if not export_path.exists():
        log.error("File not found: %s", export_path)
        sys.exit(1)

    refresh_unique_artwork()
    illus_lookup = load_illustration_lookup()
    entries = get_labelled_entries(export_path)
    log.info("Found %d labelled entries", len(entries))

    cards = []
    unresolved = []
    for task in entries:
        image = task["data"].get("image", "")
        uuid = extract_uuid(image)
        raw_name = Path(image).stem.rsplit("_", 1)[0].replace("_", " ") if image else "<unknown>"
        if not uuid:
            unresolved.append((raw_name, "<no uuid>"))
            continue
        card = illus_lookup.get(uuid)
        if not card:
            unresolved.append((raw_name, uuid))
            continue
        url = f"https://tagger.scryfall.com/card/{card['set']}/{card['collector_number']}"
        cards.append((card["name"], card["set"], card["collector_number"], url))

    if unresolved:
        log.warning("%d entries could not be resolved to a card:", len(unresolved))
        for name, uuid in unresolved:
            log.warning("  %s  (%s)", name, uuid)

    total = len(cards)
    for i, (name, set_code, number, url) in enumerate(cards, 1):
        print(f"\n[{i}/{total}] {name}  ({set_code}/{number})")
        print(f"  {url}")
        webbrowser.open(url)

        if i < total:
            try:
                inp = input("  Enter = next  |  q = quit: ").strip().lower()
                if inp == "q":
                    print("Stopped early.")
                    break
            except (KeyboardInterrupt, EOFError):
                print("\nStopped.")
                break
        else:
            print("  Done — last card opened.")


if __name__ == "__main__":
    main()
