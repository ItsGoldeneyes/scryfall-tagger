"""
Push model predictions from predictions.csv into Label Studio as pre-annotations.

Each task gets a prediction with a `score` field equal to the model's confidence.
In the Label Studio Data Manager you can then sort by "Prediction score" ascending
to surface the images the model is least sure about — label those first.

If a task already has a prediction from this model version, it is skipped.
Pass --overwrite to replace existing predictions instead.

Usage:
    python import_predictions.py
    python import_predictions.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

from _auth import make_session

load_dotenv(override=True)

log = logging.getLogger(__name__)

LS_URL = os.environ["LABEL_STUDIO_URL"]
POSITIVE_LABEL = os.environ.get("POSITIVE_LABEL", "Yes")
CHOICE_FROM_NAME = os.environ.get("LS_CHOICE_FROM_NAME", "choice")
CHOICE_TO_NAME = os.environ.get("LS_CHOICE_TO_NAME", "image")
MODEL_VERSION = "resnet18_v1"

PREDICTIONS_PATH = Path(__file__).resolve().parent / "predictions.csv"

# Maps our internal labels back to Label Studio choice values
LABEL_MAP = {"yes": POSITIVE_LABEL, "no": "No"}


def fetch_already_imported(session: requests.Session) -> set[int]:
    """Return task IDs that already have a prediction from MODEL_VERSION."""
    imported: set[int] = set()
    page = 1
    while True:
        resp = session.get(
            f"{LS_URL}/api/predictions/",
            params={"model_version": MODEL_VERSION, "page_size": 1000, "page": page},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", data) if isinstance(data, dict) else data
        if not results:
            break
        for pred in results:
            imported.add(pred["task"])
        if not (isinstance(data, dict) and data.get("next")):
            break
        page += 1
    return imported


def delete_existing_predictions(session: requests.Session, task_id: int) -> None:
    resp = session.get(f"{LS_URL}/api/predictions/", params={"task": task_id}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    predictions = data.get("results", data) if isinstance(data, dict) else data
    for pred in predictions:
        if pred.get("model_version") == MODEL_VERSION:
            session.delete(f"{LS_URL}/api/predictions/{pred['id']}/", timeout=30)


def push_prediction(
    session: requests.Session,
    task_id: int,
    label: str,
    confidence: float,
    overwrite: bool,
) -> None:
    if overwrite:
        delete_existing_predictions(session, task_id)

    payload = {
        "task": task_id,
        "result": [
            {
                "type": "choices",
                "from_name": CHOICE_FROM_NAME,
                "to_name": CHOICE_TO_NAME,
                "value": {"choices": [label]},
            }
        ],
        "score": confidence,
        "model_version": MODEL_VERSION,
    }
    resp = session.post(f"{LS_URL}/api/predictions/", json=payload, timeout=30)
    resp.raise_for_status()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing predictions for this model version before importing",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not PREDICTIONS_PATH.exists():
        log.error("predictions.csv not found — run predict.py first.")
        return

    with open(PREDICTIONS_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    session = make_session()

    if not args.overwrite:
        log.info("Fetching already-imported predictions to skip...")
        already_imported = fetch_already_imported(session)
        log.info("Found %d already imported — skipping them", len(already_imported))
        rows = [r for r in rows if int(r["task_id"]) not in already_imported]

    log.info("Importing %d predictions (overwrite=%s)...", len(rows), args.overwrite)

    def process(row: dict) -> bool:
        label = LABEL_MAP.get(row["predicted_label"], row["predicted_label"])
        for attempt in range(3):
            try:
                push_prediction(session, int(row["task_id"]), label, float(row["confidence"]), args.overwrite)
                return True
            except Exception as exc:
                if attempt == 2:
                    log.warning("Task %s failed after 3 attempts: %s", row["task_id"], exc)
                    return False
                time.sleep(2 ** attempt)
        return False

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(process, row): row for row in rows}
        with tqdm(total=len(rows), desc="Importing") as bar:
            for future in as_completed(futures):
                if future.result():
                    ok += 1
                else:
                    fail += 1
                bar.update(1)

    log.info("Done.  ok=%d  failed=%d", ok, fail)
    log.info(
        'In Label Studio Data Manager, sort by "Prediction score" ascending '
        "to see the most uncertain images first."
    )


if __name__ == "__main__":
    main()
