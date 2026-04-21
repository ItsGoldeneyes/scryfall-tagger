# magic_classifier
A utility for classifying Magic: The Gathering cards

## CLI

Run the command-line entry point with:

```bash
python card_classifier.py --help
```

The `-d` / `--data` argument points at the data directory and defaults to `./data`.
Use `-r` / `--refresh` to download the Scryfall bulk data and image cache into that directory.
Use `-c` / `--classify-image` to classify a single image and write tag IDs into the mappings file.
Use `--classify-all` to classify all `*.jpg` files in `data/art`.
Use `-m` / `--mappings` to override the default mappings filename (`color_mappings.v1.json`).
Use `--dry-run` with classification flags to preview predictions without writing mapping changes.


## Requirements
- Python 3.9 or higher
- OpenCV
- requests
- numpy