#!/bin/bash
# Kaggle dataset setup helper for Bibliophile RAG

set -e

echo "=================================================="
echo "Kaggle Dataset Setup for Bibliophile RAG"
echo "=================================================="

# Check if Kaggle dependencies are installed
echo ""
echo "Checking dependencies..."
python3 -c "import kaggle; print('✓ kaggle installed')" 2>/dev/null || {
    echo "✗ kaggle not installed"
    echo "Installing: pip install kaggle pandas"
    pip install kaggle pandas
}

python3 -c "import pandas; print('✓ pandas installed')" 2>/dev/null || {
    echo "Installing pandas..."
    pip install pandas
}

# Check Kaggle credentials
echo ""
echo "Checking Kaggle setup..."

KAGGLE_DIR="$HOME/.kaggle"
KAGGLE_JSON="$KAGGLE_DIR/kaggle.json"

if [ -f "$KAGGLE_JSON" ]; then
    echo "✓ Kaggle credentials found"

    # Fix permissions if needed
    PERMS=$(stat -c "%a" "$KAGGLE_JSON" 2>/dev/null || stat -f "%A" "$KAGGLE_JSON" 2>/dev/null)
    if [ "$PERMS" != "600" ]; then
        echo "Fixing permissions..."
        chmod 600 "$KAGGLE_JSON"
        echo "✓ Fixed"
    fi
else
    echo "✗ Kaggle credentials not found!"
    echo ""
    echo "To set up Kaggle:"
    echo "1. Go to: https://www.kaggle.com/settings/account"
    echo "2. Click 'Create New API Token'"
    echo "3. This downloads kaggle.json"
    echo "4. Run:"
    echo "   mkdir -p $KAGGLE_DIR"
    echo "   mv ~/Downloads/kaggle.json $KAGGLE_DIR/"
    echo "   chmod 600 $KAGGLE_JSON"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# Run verification
echo ""
echo "Verifying Kaggle setup..."
python3 kaggle_utils.py --check

echo ""
echo "=================================================="
echo "✓ Kaggle setup complete!"
echo "=================================================="
echo ""
echo "Next steps:"
echo ""
echo "1. List available datasets:"
echo "   python kaggle_utils.py --list"
echo ""
echo "2. Download a dataset:"
echo "   python kaggle_utils.py --dataset books --output ./goodreads_data"
echo ""
echo "3. Process CSV to text files:"
echo "   python kaggle_utils.py --process-csv ./goodreads_data/books.csv"
echo ""
echo "4. Ingest books:"
echo "   python app.py ingest ./processed_books ./chroma_db"
echo ""
echo "5. Query:"
echo "   python app.py query ./chroma_db 'science fiction' -k 5"
echo ""
echo "For more details, see: KAGGLE_DATASETS.md"
echo "=================================================="

