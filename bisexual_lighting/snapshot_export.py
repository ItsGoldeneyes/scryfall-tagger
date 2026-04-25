"""
Download a Label Studio snapshot export for the configured project.

Uses the Snapshot API (3-step: create → poll → download) to get a JSON
export of all tasks including annotations and predictions.

Usage:
  python -m bisexual_lighting.snapshot_export
  python -m bisexual_lighting.snapshot_export --out my_export.json
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from ._auth import make_session

load_dotenv(override=True)

log = logging.getLogger(__name__)

LS_URL = os.environ["LABEL_STUDIO_URL"]
PROJECT_ID = os.environ["LABEL_STUDIO_PROJECT_ID"]

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "export.json"
POLL_INTERVAL = 3  # seconds between status checks
MAX_WAIT = 300     # give up after 5 minutes


def create_snapshot(session: requests.Session) -> int:
    resp = session.post(
        f"{LS_URL}/api/projects/{PROJECT_ID}/exports/",
        json={},
        timeout=120,
    )
    resp.raise_for_status()
    snapshot_id = resp.json()["id"]
    log.info("Snapshot created: id=%s", snapshot_id)
    return snapshot_id


def wait_for_snapshot(session: requests.Session, snapshot_id: int) -> None:
    deadline = time.monotonic() + MAX_WAIT
    while time.monotonic() < deadline:
        resp = session.get(
            f"{LS_URL}/api/projects/{PROJECT_ID}/exports/{snapshot_id}",
            timeout=30,
        )
        resp.raise_for_status()
        status = resp.json().get("status")
        log.info("Snapshot status: %s", status)
        if status == "completed":
            return
        if status == "failed":
            raise RuntimeError(f"Snapshot {snapshot_id} failed on the server.")
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Snapshot did not complete within {MAX_WAIT}s.")


def download_snapshot(session: requests.Session, snapshot_id: int, out_path: Path) -> None:
    resp = session.get(
        f"{LS_URL}/api/projects/{PROJECT_ID}/exports/{snapshot_id}/download",
        params={"exportType": "JSON"},
        timeout=360,
        stream=True,
    )
    resp.raise_for_status()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
    log.info("Saved export to %s (%d bytes)", out_path, out_path.stat().st_size)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Where to write the JSON export (default: export.json in repo root)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    session = make_session()

    snapshot_id = create_snapshot(session)
    wait_for_snapshot(session, snapshot_id)
    download_snapshot(session, snapshot_id, args.out)
    log.info("Done. Pass %s to export_labels.py or predict.py.", args.out)


if __name__ == "__main__":
    main()
