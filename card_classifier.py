"""CLI entry point for the Scryfall tagger project.

Supports refreshing Scryfall bulk data, downloading card art, and classifying
card images using HSV-based colour matching against a JSON mapping file.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.scryfall_tools import refresh_scryfall_data
from tools.tagger_tools import create_default_mapping, save_mapping_file, tag_and_update


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Tool to classify MTG cards according to Scryfall tags",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--data",
        action="store",
        default=None,
        help="Path to the data directory (default: $DATA_DIR env var, then ./data)",
    )
    parser.add_argument(
        "-r",
        "--refresh",
        action="store_true",
        help="Refresh Scryfall bulk data and image downloads",
    )
    parser.add_argument(
        "-c",
        "--classify-image",
        action="store",
        default=None,
        help="Classify a single image file (filename or path)",
    )
    parser.add_argument(
        "--classify-all",
        action="store_true",
        help="Classify all jpg images in the art directory",
    )
    parser.add_argument(
        "-m",
        "--mappings",
        action="store",
        default="color_mappings.v1.json",
        help="Mapping filename in the data directory or absolute path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run classification without writing mapping changes to disk",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-tag HSV scores alongside classification results",
    )
    return parser


def prepare_data_directory(data_path: Path) -> None:
    """Create the data directory structure used by the CLI."""
    data_path.mkdir(parents=True, exist_ok=True)
    (data_path / "art").mkdir(parents=True, exist_ok=True)


def resolve_mapping_path(data_path: Path, mapping_arg: str) -> Path:
    """Resolve mapping path from CLI argument."""
    candidate = Path(mapping_arg).expanduser()
    if candidate.is_absolute():
        return candidate
    return data_path / candidate


def resolve_single_image_path(data_path: Path, image_arg: str) -> Path:
    """Resolve image path from CLI argument."""
    candidate = Path(image_arg).expanduser()
    if candidate.is_absolute() and candidate.exists():
        return candidate
    if candidate.exists():
        return candidate.resolve()

    by_art = data_path / "art" / image_arg
    if by_art.exists():
        return by_art

    raise FileNotFoundError("Image not found: %s" % image_arg)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    import os
    data_path = Path(args.data or os.environ.get("DATA_DIR") or "./data").expanduser().resolve()
    prepare_data_directory(data_path)
    mapping_path = resolve_mapping_path(data_path, args.mappings)

    logging.info("Using data directory: %s", data_path)
    if args.refresh:
        refresh_scryfall_data(data_path)

    images_to_classify: List[Path] = []
    if args.classify_image:
        images_to_classify.append(resolve_single_image_path(data_path, args.classify_image))
    if args.classify_all:
        images_to_classify.extend(sorted((data_path / "art").glob("*.jpg")))

    if images_to_classify:
        if not mapping_path.exists():
            logging.info("Mapping file not found; creating default at %s", mapping_path)
            save_mapping_file(mapping_path, create_default_mapping())

        results = tag_and_update(images_to_classify, mapping_path, dry_run=args.dry_run)
        for result in results:
            tag_summary = ", ".join(
                "%s (%.4f)" % (t["id"], t["confidence"]) for t in result["tags"]
            )
            logging.info("Classified %s => [%s]", result["image_file"], tag_summary)
            if args.verbose:
                for tag_id, score in sorted(result["scores"].items(), key=lambda x: x[1], reverse=True):
                    logging.info("  %-12s %.4f", tag_id, score)
    elif not args.refresh:
        logging.info("No action requested. Use --refresh, --classify-image, or --classify-all.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
