# scryfall-tagger

Utilities for a bisexual lighting labeling workflow on Magic: The Gathering art.

## Repository Structure

- Root scripts: Label Studio export/import helpers plus train/predict pipeline.
- tools/: shared Scryfall utility helpers.
- data/: local data storage (`.gitkeep` is committed; generated data is ignored).

## Main Workflow

1. Log in to Label Studio and save a session cookie:

```bash
python get_ls_token.py
```

2. Export project tasks from Label Studio (or use an existing export):

```bash
python snapshot_export.py
```

3. Download labeled images into local training folders:

```bash
python export_labels.py export.json
```

4. Train the classifier:

```bash
python train.py
```

5. Score unlabeled tasks with uncertainty ranking:

```bash
python predict.py export.json
```

6. Import predictions back into Label Studio:

```bash
python import_predictions.py
```

Optional review helper:

```bash
python review.py export.json
```

## Requirements

- Python 3.9 or higher
- See `requirements.txt`