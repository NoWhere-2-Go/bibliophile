# Kaggle Integration - Quick Reference

## Installation (2 minutes)

```bash
# 1. Update dependencies
pip install kaggle pandas

# 2. Get API credentials
# Go to: https://www.kaggle.com/settings/account
# Click: Create New API Token (downloads kaggle.json)
# Then run:
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 3. Verify
python kaggle_utils.py --check
```

## Usage (One Command Each)

### View Available Datasets
```bash
python kaggle_utils.py --list
```

### Download a Dataset
```bash
# Quick test (100 books, 1min)
python kaggle_utils.py --dataset amazon-books --output ./test

# Recommended (11k books, 5min)
python kaggle_utils.py --dataset books --output ./goodreads

# Large (100k books, 30min)
python kaggle_utils.py --dataset goodreads-100k --output ./large

# Classics (70k full texts, 1hour)
python kaggle_utils.py --dataset gutenberg --output ./gutenberg
```

### Convert CSV to Text Files
```bash
python kaggle_utils.py --process-csv ./goodreads/books.csv
```

### Ingest into Vector Store
```bash
python app.py ingest ./processed_books ./chroma_db
```

### Query
```bash
python app.py query ./chroma_db "science fiction" -k 10
```

## Complete 5-Minute Workflow

```bash
# 1. Download small dataset (2 min)
python kaggle_utils.py --dataset amazon-books --output ./test_data

# 2. Process CSV (1 min)
python kaggle_utils.py --process-csv ./test_data/books_amazon.csv

# 3. Ingest (2 min)
python app.py ingest ./processed_books ./test_index

# 4. Query
python app.py query ./test_index "bestseller" -k 5
```

## Dataset Comparison

| Dataset | Size | Books | Time |
|---------|------|-------|------|
| amazon-books | 1MB | 100 | 1 min |
| goodreads | 50MB | 11k | 5 min |
| goodreads-100k | 500MB | 100k | 30 min |
| gutenberg | 5GB | 70k | 1 hour |

## Troubleshooting

**Kaggle API not found?**
```bash
python kaggle_utils.py --check
# Follow instructions to set up
```

**Dataset not found?**
```bash
python kaggle_utils.py --list
# Use correct ID from list
```

**Want different dataset?**
```bash
python kaggle_utils.py --dataset-id owner/dataset-name --output ./data
```

## Commands

```bash
# Setup
chmod +x setup_kaggle.sh
./setup_kaggle.sh

# Verify
python kaggle_utils.py --check

# List
python kaggle_utils.py --list

# Download
python kaggle_utils.py --dataset books --output ./data

# Process
python kaggle_utils.py --process-csv ./data/books.csv

# Ingest
python app.py ingest ./processed_books ./chroma_db -v

# Query
python app.py query ./chroma_db "your search" -k 10
```

## Files

- `kaggle_utils.py` - Download & process datasets
- `setup_kaggle.sh` - Automated setup
- `KAGGLE_DATASETS.md` - Full documentation
- `KAGGLE_INTEGRATION.md` - Integration guide
- `requirements.txt` - Updated with kaggle, pandas

## Next Steps

1. Run `python kaggle_utils.py --check`
2. Download a dataset with `--dataset`
3. Process with `--process-csv`
4. Ingest with `python app.py ingest`
5. Query with `python app.py query`

See `KAGGLE_INTEGRATION.md` for detailed guide.

