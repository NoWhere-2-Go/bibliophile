#!/bin/bash
# Setup guide for Bibliophile RAG - Phase 1 Offline Pipeline

set -e

echo "================================================"
echo "Bibliophile RAG - Phase 1 Setup"
echo "================================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $PYTHON_VERSION found"

# Check if venv exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

echo "✓ Virtual environment ready"

# Activate venv
source .venv/bin/activate
echo "✓ Activated virtual environment"

# Upgrade pip
echo "📥 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "================================================"
echo "Dependencies installed successfully!"
echo "================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Start Ollama (if not already running):"
echo "   $ ollama serve"
echo ""
echo "2. In a new terminal, pull the embedding model:"
echo "   $ ollama pull nomic-embed-text"
echo ""
echo "3. Test the embedding service:"
echo "   $ curl -X POST 'http://localhost:11434/api/embed' \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"model\":\"nomic-embed-text\",\"input\":[\"test\"]}'"
echo ""
echo "4. Ingest sample books:"
echo "   $ python app.py ingest ./data ./chroma_db"
echo ""
echo "5. Query the index:"
echo "   $ python app.py query ./chroma_db 'books about space' -k 5"
echo ""
echo "For more help:"
echo "   $ python app.py --help"

