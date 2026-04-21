"""Simple HSV-based image tagger backed by mapping definitions."""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2 as cv
import numpy as np
from tqdm import tqdm


UUID_RE = re.compile(
    r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)


def load_mapping_file(mapping_path: Path) -> Dict:
    """Load the mapping file from disk."""
    with mapping_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_mapping_file(mapping_path: Path, data: Dict) -> None:
    """Write mapping data to disk."""
    with mapping_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _normalize_h_ranges(h_data: List) -> List[Tuple[int, int]]:
    """Convert h field into a list of hue ranges."""
    if len(h_data) == 2 and isinstance(h_data[0], int):
        return [(int(h_data[0]), int(h_data[1]))]
    ranges: List[Tuple[int, int]] = []
    for pair in h_data:
        ranges.append((int(pair[0]), int(pair[1])))
    return ranges


def _band_match_ratio(hsv_image: np.ndarray, band: Dict) -> float:
    """Return the fraction of pixels matching a single HSV band definition."""
    h_data = band.get("h")
    s_data = band.get("s", [0, 255])
    v_data = band.get("v", [0, 255])
    if h_data is None:
        return 0.0

    h_channel = hsv_image[:, :, 0]
    s_channel = hsv_image[:, :, 1]
    v_channel = hsv_image[:, :, 2]

    s_mask = (s_channel >= int(s_data[0])) & (s_channel <= int(s_data[1]))
    v_mask = (v_channel >= int(v_data[0])) & (v_channel <= int(v_data[1]))

    hue_masks = []
    for h_min, h_max in _normalize_h_ranges(h_data):
        hue_masks.append((h_channel >= h_min) & (h_channel <= h_max))

    if not hue_masks:
        return 0.0

    hue_mask = hue_masks[0]
    for additional in hue_masks[1:]:
        hue_mask = hue_mask | additional

    final_mask = hue_mask & s_mask & v_mask
    total = int(hsv_image.shape[0] * hsv_image.shape[1])
    if total == 0:
        return 0.0
    return int(np.count_nonzero(final_mask)) / float(total)


def _tag_match_ratio(hsv_image: np.ndarray, tag_def: Dict) -> float:
    """Return the fraction of pixels that match one tag definition."""
    return _band_match_ratio(hsv_image, tag_def.get("data", {}))


def _compound_match_ratio(hsv_image: np.ndarray, tag_def: Dict) -> float:
    """Return compound score requiring co-presence of multiple hue bands.

    Two modes, selected by the keys present in ``data``:

    ``bands`` (list)  — legacy single-combination mode.
        Score = min(ratio per band).  All bands must be present.

    ``combinations`` (list of lists)  — multi-combination mode.
        Each combination is a list of band dicts.  Score = max over
        combinations of min-within-combination, so the tag fires if *any*
        recognised colour pair is co-present.

    An optional ``min_saturation_mean`` gate short-circuits to 0.0 when the
    image is too desaturated to plausibly exhibit the lighting effect.
    """
    data = tag_def.get("data", {})

    min_sat = data.get("min_saturation_mean")
    if min_sat is not None:
        if float(np.mean(hsv_image[:, :, 1])) < float(min_sat):
            return 0.0

    # Legacy single-combination mode
    bands = data.get("bands")
    if bands is not None:
        ratios = [_band_match_ratio(hsv_image, band) for band in bands]
        return min(ratios) if ratios else 0.0

    # Multi-combination mode
    combinations = data.get("combinations", [])
    if not combinations:
        return 0.0

    best = 0.0
    for combo in combinations:
        ratios = [_band_match_ratio(hsv_image, band) for band in combo]
        best = max(best, min(ratios) if ratios else 0.0)
    return best


def tag_image_file(image_path: Path, mapping_data: Dict) -> Tuple[List[str], float, Dict[str, float]]:
    """Tag one image and return tag IDs ranked by pixel coverage."""
    # cv.imread fails silently on Windows paths with non-ASCII characters.
    # Reading via numpy bytes sidesteps that limitation.
    image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv.imdecode(image_bytes, cv.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unable to read image: %s" % image_path)

    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    tag_defs = mapping_data.get("tag_definitions", [])

    colour_scores: Dict[str, float] = {}
    compound_scores: Dict[str, float] = {}
    compound_thresholds: Dict[str, float] = {}

    for tag_def in tag_defs:
        if not tag_def.get("enabled", True):
            continue
        tag_type = tag_def.get("type")
        tag_id = tag_def.get("id")
        if not tag_id:
            continue

        if tag_type == "compound":
            compound_scores[tag_id] = _compound_match_ratio(hsv_image, tag_def)
            compound_thresholds[tag_id] = float(tag_def.get("min_confidence", 0.04))
        elif tag_type == "colour":
            colour_scores[tag_id] = _tag_match_ratio(hsv_image, tag_def)

    scores = {**colour_scores, **compound_scores}

    if not colour_scores:
        fallback = mapping_data.get("fallback", {}).get("id", "unknown")
        return [fallback], 0.0, {}

    ranked_colours = sorted(colour_scores.items(), key=lambda x: x[1], reverse=True)
    top_id, top_score = ranked_colours[0]

    selected = [
        {"id": tag_id, "confidence": round(score, 4)}
        for tag_id, score in ranked_colours[:5]
        if score > 0.0
    ]

    for tag_id, score in compound_scores.items():
        if score >= compound_thresholds[tag_id]:
            selected.append({"id": tag_id, "confidence": round(score, 4)})

    return selected, float(top_score), scores


def _extract_illustration_id(image_file: str) -> Optional[str]:
    """Extract illustration id from an image filename when present."""
    match = UUID_RE.search(image_file)
    if not match:
        return None
    return match.group(1)


def upsert_art_mapping(mapping_data: Dict, image_file: str, ranked_tags: List[Dict], source: str = "auto") -> None:
    """Insert or update one art mapping entry by image file."""
    art_mappings = mapping_data.setdefault("art_mappings", [])

    target = None
    for entry in art_mappings:
        if entry.get("image_file") == image_file:
            target = entry
            break

    if target is None:
        illustration_id = _extract_illustration_id(image_file)
        card_name = image_file
        if illustration_id:
            card_name = image_file.replace("_%s.jpg" % illustration_id, "").replace("_", " ")

        target = {
            "card_name": card_name,
            "illustration_id": illustration_id,
            "image_file": image_file,
            "tags": [],
            "source": source,
        }
        art_mappings.append(target)

    target["tags"] = ranked_tags
    target["source"] = source


def create_default_mapping() -> Dict:
    """Return a default mapping dict covering the full visible spectrum plus neutrals.

    Hue ranges follow OpenCV convention (0-180 for 0-360°).
    Each chromatic colour has dark / mid / light variants distinguished by V (brightness)
    and S (saturation). Neutrals are split by V into five steps.

    Dark   = V  15-105, typically high S
    Mid    = V  80-210, medium-high S
    Light  = V 155-255, lower S (pastel / washed-out)
    """
    return {
        "tag_definitions": [
            # ------------------------------------------------------------------ #
            # Neutrals (no hue — distinguished purely by V)                       #
            # ------------------------------------------------------------------ #
            {
                "id": "black",
                "label": "Black",
                "type": "colour",
                "enabled": True,
                "data": {"h": [0, 180], "s": [0, 255], "v": [0, 40]},
            },
            {
                "id": "dark-grey",
                "label": "Dark Grey",
                "type": "colour",
                "enabled": True,
                "data": {"h": [0, 180], "s": [0, 60], "v": [41, 105]},
            },
            {
                "id": "grey",
                "label": "Grey",
                "type": "colour",
                "enabled": True,
                "data": {"h": [0, 180], "s": [0, 60], "v": [106, 165]},
            },
            {
                "id": "light-grey",
                "label": "Light Grey",
                "type": "colour",
                "enabled": True,
                "data": {"h": [0, 180], "s": [0, 60], "v": [166, 210]},
            },
            {
                "id": "white",
                "label": "White",
                "type": "colour",
                "enabled": True,
                "data": {"h": [0, 180], "s": [0, 45], "v": [211, 255]},
            },
            # ------------------------------------------------------------------ #
            # Browns / Tans (warm earth tones, low-mid V, moderate S)             #
            # ------------------------------------------------------------------ #
            {
                "id": "dark-brown",
                "label": "Dark Brown",
                "type": "colour",
                "enabled": True,
                "data": {"h": [5, 20], "s": [90, 255], "v": [15, 80]},
            },
            {
                "id": "brown",
                "label": "Brown",
                "type": "colour",
                "enabled": True,
                "data": {"h": [8, 22], "s": [70, 200], "v": [60, 145]},
            },
            {
                "id": "tan",
                "label": "Tan / Beige",
                "type": "colour",
                "enabled": True,
                "data": {"h": [12, 28], "s": [30, 130], "v": [130, 215]},
            },
            # ------------------------------------------------------------------ #
            # Reds  H: 0-10 and 165-180                                           #
            # ------------------------------------------------------------------ #
            {
                "id": "dark-red",
                "label": "Dark Red / Crimson",
                "type": "colour",
                "enabled": True,
                "data": {"h": [[0, 10], [165, 180]], "s": [120, 255], "v": [20, 105]},
            },
            {
                "id": "red",
                "label": "Red",
                "type": "colour",
                "enabled": True,
                "data": {"h": [[0, 10], [165, 180]], "s": [90, 255], "v": [85, 225]},
            },
            {
                "id": "light-red",
                "label": "Light Red / Rose",
                "type": "colour",
                "enabled": True,
                "data": {"h": [[0, 15], [155, 180]], "s": [35, 145], "v": [175, 255]},
            },
            # ------------------------------------------------------------------ #
            # Oranges  H: 10-22                                                   #
            # ------------------------------------------------------------------ #
            {
                "id": "burnt-orange",
                "label": "Burnt Orange",
                "type": "colour",
                "enabled": True,
                "data": {"h": [10, 22], "s": [140, 255], "v": [55, 165]},
            },
            {
                "id": "orange",
                "label": "Orange",
                "type": "colour",
                "enabled": True,
                "data": {"h": [10, 22], "s": [100, 255], "v": [120, 255]},
            },
            {
                "id": "peach",
                "label": "Peach / Light Orange",
                "type": "colour",
                "enabled": True,
                "data": {"h": [8, 25], "s": [30, 140], "v": [180, 255]},
            },
            # ------------------------------------------------------------------ #
            # Yellows / Golds  H: 20-38                                           #
            # ------------------------------------------------------------------ #
            {
                "id": "gold",
                "label": "Gold / Amber",
                "type": "colour",
                "enabled": True,
                "data": {"h": [20, 35], "s": [110, 255], "v": [80, 210]},
            },
            {
                "id": "yellow",
                "label": "Yellow",
                "type": "colour",
                "enabled": True,
                "data": {"h": [22, 38], "s": [70, 255], "v": [160, 255]},
            },
            {
                "id": "cream",
                "label": "Cream / Light Yellow",
                "type": "colour",
                "enabled": True,
                "data": {"h": [15, 38], "s": [10, 85], "v": [200, 255]},
            },
            # ------------------------------------------------------------------ #
            # Greens  H: 35-85                                                    #
            # ------------------------------------------------------------------ #
            {
                "id": "dark-green",
                "label": "Dark Green / Forest",
                "type": "colour",
                "enabled": True,
                "data": {"h": [38, 85], "s": [90, 255], "v": [15, 105]},
            },
            {
                "id": "olive",
                "label": "Olive",
                "type": "colour",
                "enabled": True,
                "data": {"h": [22, 45], "s": [60, 200], "v": [45, 150]},
            },
            {
                "id": "green",
                "label": "Green",
                "type": "colour",
                "enabled": True,
                "data": {"h": [35, 85], "s": [80, 255], "v": [80, 210]},
            },
            {
                "id": "light-green",
                "label": "Light Green / Mint",
                "type": "colour",
                "enabled": True,
                "data": {"h": [35, 85], "s": [30, 145], "v": [155, 255]},
            },
            # ------------------------------------------------------------------ #
            # Teals / Cyans  H: 80-100                                            #
            # ------------------------------------------------------------------ #
            {
                "id": "teal",
                "label": "Teal / Dark Cyan",
                "type": "colour",
                "enabled": True,
                "data": {"h": [80, 100], "s": [90, 255], "v": [25, 145]},
            },
            {
                "id": "cyan",
                "label": "Cyan",
                "type": "colour",
                "enabled": True,
                "data": {"h": [80, 100], "s": [70, 255], "v": [100, 255]},
            },
            {
                "id": "light-cyan",
                "label": "Light Cyan",
                "type": "colour",
                "enabled": True,
                "data": {"h": [80, 100], "s": [20, 115], "v": [165, 255]},
            },
            # ------------------------------------------------------------------ #
            # Blues  H: 100-130                                                   #
            # ------------------------------------------------------------------ #
            {
                "id": "navy",
                "label": "Navy / Dark Blue",
                "type": "colour",
                "enabled": True,
                "data": {"h": [100, 130], "s": [110, 255], "v": [15, 105]},
            },
            {
                "id": "blue",
                "label": "Blue",
                "type": "colour",
                "enabled": True,
                "data": {"h": [100, 130], "s": [80, 255], "v": [85, 210]},
            },
            {
                "id": "light-blue",
                "label": "Light Blue / Sky Blue",
                "type": "colour",
                "enabled": True,
                "data": {"h": [100, 130], "s": [20, 135], "v": [155, 255]},
            },
            # ------------------------------------------------------------------ #
            # Purples  H: 128-160                                                 #
            # ------------------------------------------------------------------ #
            {
                "id": "dark-purple",
                "label": "Dark Purple / Indigo",
                "type": "colour",
                "enabled": True,
                "data": {"h": [128, 158], "s": [95, 255], "v": [18, 110]},
            },
            {
                "id": "purple",
                "label": "Purple",
                "type": "colour",
                "enabled": True,
                "data": {"h": [128, 160], "s": [55, 255], "v": [55, 205]},
            },
            {
                "id": "lavender",
                "label": "Lavender",
                "type": "colour",
                "enabled": True,
                "data": {"h": [128, 160], "s": [18, 110], "v": [155, 255]},
            },
            # ------------------------------------------------------------------ #
            # Pinks / Magentas  H: 148-180                                        #
            # ------------------------------------------------------------------ #
            {
                "id": "dark-pink",
                "label": "Dark Pink / Deep Magenta",
                "type": "colour",
                "enabled": True,
                "data": {"h": [148, 175], "s": [110, 255], "v": [55, 175]},
            },
            {
                "id": "pink",
                "label": "Pink / Magenta",
                "type": "colour",
                "enabled": True,
                "data": {"h": [148, 175], "s": [65, 255], "v": [110, 255]},
            },
            {
                "id": "light-pink",
                "label": "Light Pink",
                "type": "colour",
                "enabled": True,
                "data": {"h": [145, 180], "s": [15, 105], "v": [185, 255]},
            },
            # ------------------------------------------------------------------ #
            # Compound tags (require co-presence of multiple hue bands)           #
            # ------------------------------------------------------------------ #
            {
                "id": "bisexual-lighting",
                "label": "Bisexual Lighting",
                "type": "compound",
                "enabled": True,
                "min_confidence": 0.04,
                "data": {
                    "min_saturation_mean": 70,
                    "bands": [
                        # Pink / magenta — H capped at 172 to exclude the near-red tail (H>172 ≈ >344°)
                        {"h": [148, 172], "s": [80, 255], "v": [50, 255]},
                        # Blue / blue-purple side (cool end)
                        {"h": [100, 130], "s": [80, 255], "v": [50, 220]},
                    ],
                },
            },
        ],
        "fallback": {"id": "grey"},
        "art_mappings": [],
    }


def tag_and_update(image_paths: List[Path], mapping_path: Path, dry_run: bool = False) -> List[Dict[str, object]]:
    """Tag images and optionally write updated mappings back to disk.

    Handles KeyboardInterrupt gracefully: partial results are saved before exit.
    """
    mapping_data = load_mapping_file(mapping_path)
    results: List[Dict[str, object]] = []
    interrupted = False

    try:
        for image_path in tqdm(image_paths, desc="Classifying", unit="img"):
            try:
                selected, confidence, scores = tag_image_file(image_path, mapping_data)
            except ValueError as exc:
                logging.warning("Skipping %s: %s", image_path.name, exc)
                continue
            upsert_art_mapping(mapping_data, image_path.name, selected, source="auto")
            results.append(
                {
                    "image_file": image_path.name,
                    "tags": selected,
                    "scores": {k: round(v, 4) for k, v in scores.items()},
                }
            )
    except KeyboardInterrupt:
        interrupted = True
        logging.warning("Interrupted — saving %d result(s) classified so far.", len(results))

    if dry_run:
        logging.info("Dry run enabled: no mapping changes were written")
    else:
        save_mapping_file(mapping_path, mapping_data)
        logging.info("Updated mappings saved to %s", mapping_path)

    if interrupted:
        raise SystemExit(1)

    return results
