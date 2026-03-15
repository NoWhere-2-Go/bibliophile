"""Kaggle dataset utilities for Bibliophile RAG.

Supports downloading and processing popular book datasets from Kaggle.
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

POPULAR_DATASETS = {
    "books": {
        "id": "jealousleopard/goodreadsbooks",
        "format": "csv",
        "description": "GoodReads Books Dataset (~11k books with metadata)",
        "text_columns": ["book_desc", "description"],
        "metadata_columns": {"title": "title", "author": "authors", "year": "publication_date"},
    },
    "GoodReads_100k_books": {
        "id": "mdhamani/goodreads-books-100k",
        "format": "csv",
        "description": "GoodReads 100k Books Dataset with descriptions",
        "text_columns": ["desc"],
        "metadata_columns": {"title": "title", "author": "author", "genre": "genre", "isbn": "isbn", "pages": "pages"},
    }
}


def check_kaggle_setup() -> bool:
    """Check if Kaggle API is set up correctly."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        logger.error("Kaggle API credentials not found!")
        logger.error(f"Expected file: {kaggle_json}")
        logger.error("\nTo set up:")
        logger.error("1. Go to https://www.kaggle.com/settings/account")
        logger.error("2. Click 'Create New API Token'")
        logger.error("3. This downloads kaggle.json")
        logger.error(f"4. Move it to: {kaggle_dir}")
        logger.error("5. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    # Check permissions
    perms = oct(kaggle_json.stat().st_mode)[-3:]
    if perms != "600":
        logger.warning(f"Kaggle credentials have loose permissions: {perms}")
        logger.warning("Running: chmod 600 ~/.kaggle/kaggle.json")
        os.chmod(kaggle_json, 0o600)
    
    logger.info("✓ Kaggle API credentials found")
    return True


def download_kaggle_dataset(
    dataset_id: str,
    output_dir: str = "./kaggle_data",
    unzip: bool = True
) -> bool:
    """Download a dataset from Kaggle.
    
    Args:
        dataset_id: Kaggle dataset ID (e.g., "jealousleopard/goodreadsbooks")
        output_dir: Directory to download to
        unzip: Whether to unzip downloaded files
    
    Returns:
        True if successful, False otherwise
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        logger.error("kaggle package not installed. Install with: pip install kaggle")
        return False
    
    if not check_kaggle_setup():
        return False
    
    try:
        logger.info(f"Downloading dataset: {dataset_id}")
        api = KaggleApi()
        api.authenticate()
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        api.dataset_download_files(dataset_id, path=str(output_path), unzip=unzip)
        logger.info(f"✓ Downloaded to: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        return False


def list_popular_datasets():
    """Display popular datasets available on Kaggle."""
    print("\n" + "="*70)
    print("POPULAR KAGGLE BOOK DATASETS")
    print("="*70)
    
    for key, info in POPULAR_DATASETS.items():
        print(f"\n[{key}]")
        print(f"  Name: {info['description']}")
        print(f"  ID: {info['id']}")
        print(f"  Format: {info['format']}")
        print(f"  Metadata: {info['metadata_columns']}")
    
    print("\n" + "="*70)
    print("\nUsage Example:")
    print("  python ingest_kaggle.py --dataset books --output ./kaggle_data")
    print("="*70 + "\n")


def process_csv_dataset(
    csv_path: str,
    text_columns: List[str],
    metadata_columns: Dict[str, str],
    output_dir: str = "./processed_books"
) -> int:
    """Convert CSV dataset to individual text files.
    
    Args:
        csv_path: Path to CSV file
        text_columns: Columns containing book text/description
        metadata_columns: Mapping of metadata fields to column names
        output_dir: Where to save text files
    
    Returns:
        Number of files created
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas not installed. Install with: pip install pandas")
        return 0
    
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Reading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"CSV loaded: {len(df)} rows, columns: {df.columns.tolist()}")
        
        # Validate that required text columns exist
        available_text_cols = [col for col in text_columns if col in df.columns]
        if not available_text_cols:
            logger.error(f"None of the text columns {text_columns} found in CSV!")
            logger.error(f"Available columns: {df.columns.tolist()}")
            return 0
        
        logger.info(f"Using text columns: {available_text_cols}")
        
        # Validate metadata columns
        for field, col in metadata_columns.items():
            if col not in df.columns:
                logger.warning(f"Metadata column '{col}' (for {field}) not found in CSV")
        
        file_count = 0
        skipped_count = 0
        for idx, row in df.iterrows():
            # Get text content
            text_content = None
            for col in available_text_cols:
                if pd.notna(row[col]):
                    text_content = str(row[col]).strip()
                    if len(text_content) > 50:  # Minimum content length
                        break
            
            if not text_content or len(text_content) < 50:
                skipped_count += 1
                continue
            
            # Build filename with metadata
            title = row.get(metadata_columns.get("title", ""), "book")
            author = row.get(metadata_columns.get("author", ""), "unknown")
            year = row.get(metadata_columns.get("year", ""), "")
            
            # Clean strings for filename
            title = str(title).replace("/", "-")[:50].strip()
            author = str(author).replace("/", "-")[:30].strip()
            year = str(year)[:4] if year else ""
            
            if year and year.isdigit():
                filename = f"{author} - {title} ({year}).txt"
            else:
                filename = f"{author} - {title}.txt"
            
            filepath = output_path / filename
            
            # Write file with metadata header
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    if title and title != "book":
                        f.write(f"Title: {title}\n")
                    if author and author != "unknown":
                        f.write(f"Author: {author}\n")
                    if year and year.isdigit():
                        f.write(f"Year: {year}\n")
                    
                    # Write additional metadata from CSV columns
                    # Common metadata field names to check for
                    metadata_fields = {
                        "Genre": ["genre", "genres", "category", "categories"],
                        "ISBN": ["isbn", "isbn13", "isbn10"],
                        "Pages": ["pages", "page_count", "num_pages"],
                        "Rating": ["rating", "avg_rating", "average_rating"],
                        "Total Ratings": ["totalratings", "total_ratings", "num_ratings", "ratings_count"],
                        "Reviews": ["reviews", "num_reviews", "review_count", "reviews_count"],
                        "Book Format": ["bookformat", "book_format", "format"],
                        "Language": ["language", "lang"],
                        "Publisher": ["publisher"],
                        "Edition": ["edition"],
                        "Link": ["link", "url", "goodreads_link"],
                    }
                    
                    # Extract and write available metadata
                    for field_name, possible_cols in metadata_fields.items():
                        for col in possible_cols:
                            if col in df.columns and pd.notna(row[col]):
                                value = str(row[col]).strip()
                                if value and value.lower() != "nan":
                                    f.write(f"{field_name}: {value}\n")
                                    break  # Only use first found column
                    
                    f.write("\n")
                    f.write(text_content)
                
                file_count += 1
                if file_count % 100 == 0:
                    logger.info(f"Processed {file_count} books...")
            except Exception as e:
                logger.warning(f"Failed to write {filename}: {e}")
                continue
        
        logger.info(f"✓ Created {file_count} text files in {output_path}")
        logger.info(f"  Skipped {skipped_count} rows (insufficient text content)")
        return file_count
    
    except Exception as e:
        logger.error(f"Failed to process CSV: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download and process Kaggle book datasets for Bibliophile RAG"
    )
    
    parser.add_argument(
        "--dataset",
        choices=list(POPULAR_DATASETS.keys()),
        help="Popular dataset to download"
    )
    parser.add_argument(
        "--dataset-id",
        help="Custom Kaggle dataset ID (format: owner/dataset-name)"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List popular datasets"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check Kaggle setup"
    )
    parser.add_argument(
        "--output",
        default="./kaggle_data",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--process-csv",
        metavar="CSV_FILE",
        help="Process a CSV file to text files"
    )
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    if args.list:
        list_popular_datasets()
    
    elif args.check:
        if check_kaggle_setup():
            print("✓ Kaggle is properly configured!")
        else:
            print("✗ Kaggle setup is incomplete")
    
    elif args.process_csv:
        # Example: process a CSV file
        print("\nNote: This is a basic processor.")
        print("For best results, customize the text_columns and metadata_columns")
        print("in the POPULAR_DATASETS dictionary.\n")
        
        from rag.ingest import ingest_directory

        logger.info(f"args: {args}")
        if args.dataset is None:
            logger.error("Please specify a dataset with --dataset to get column info for processing.")
            exit(1)

        dataset = POPULAR_DATASETS[args.dataset]

        # Create temporary text files from CSV
        process_csv_dataset(
            args.process_csv,
            text_columns=dataset.get("text_columns", []),
            metadata_columns=dataset.get("metadata_columns", {}),
            output_dir="./processed_books"
        )
        
        logger.info("\nFinished processing CSV files.")
    
    elif args.dataset:
        dataset_info = POPULAR_DATASETS[args.dataset]
        print(f"\nDownloading: {dataset_info['description']}")
        
        if download_kaggle_dataset(dataset_info['id'], args.output):
            print(f"\n✓ Dataset downloaded to: {args.output}")
            print(f"Format: {dataset_info['format']}")
            
            if dataset_info['format'] == 'csv':
                print("\nNext step: Process CSV to text files")
                print(f"  python kaggle_utils.py --process-csv {args.output}/*.csv")
            elif dataset_info['format'] == 'txt':
                print("\nNext step: Use with ingestion")
                print(f"  python app.py ingest {args.output} ./chroma_db")
    
    elif args.dataset_id:
        print(f"\nDownloading custom dataset: {args.dataset_id}")
        if download_kaggle_dataset(args.dataset_id, args.output):
            print(f"✓ Dataset downloaded to: {args.output}")
    
    else:
        parser.print_help()

