#!/usr/bin/env python3
"""
Verification script for Bibliophile RAG Phase 1 setup.
Tests all components and displays a health report.
"""
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

def check_files():
    """Verify all required files exist."""
    logger.info("\n" + "="*60)
    logger.info("FILE STRUCTURE CHECK")
    logger.info("="*60)
    
    required_files = {
        "Core": [
            "app.py",
            "requirements.txt",
            "README.md",
            ".gitignore",
        ],
        "RAG Module": [
            "rag/__init__.py",
            "rag/ingest.py",
            "rag/embeddings.py",
            "rag/vectorstore.py",
            "rag/retriever.py",
        ],
        "Documentation": [
            "rag-architecture.html",
            ".env.example",
        ],
        "Scripts": [
            "setup.sh",
            "example.py",
        ],
        "Data": [
            "data/amazon-books/bestsellers_with_categories.csv",
            "data/goodreads/books.csv"
        ]
    }
    
    all_ok = True
    for category, files in required_files.items():
        logger.info(f"\n{category}:")
        for filename in files:
            path = Path(filename)
            if path.exists():
                size = path.stat().st_size
                logger.info(f"  ✓ {filename} ({size} bytes)")
            else:
                logger.info(f"  ✗ {filename} (MISSING)")
                all_ok = False
    
    return all_ok

def check_imports():
    """Verify all Python modules can be imported."""
    logger.info("\n" + "="*60)
    logger.info("MODULE IMPORT CHECK")
    logger.info("="*60)
    
    modules = [
        ("rag.ingest", ["ingest_directory", "chunk_text_by_tokens", "extract_book_metadata"]),
        ("rag.embeddings", ["EmbeddingModel", "batch_embed"]),
        ("rag.vectorstore", ["ChromaVectorStore"]),
        ("rag.retriever", ["Retriever"]),
    ]
    
    all_ok = True
    for module_name, exports in modules:
        try:
            module = __import__(module_name, fromlist=exports)
            logger.info(f"\n✓ {module_name}")
            for export in exports:
                if hasattr(module, export):
                    logger.info(f"    ✓ {export}")
                else:
                    logger.info(f"    ✗ {export} (NOT FOUND)")
                    all_ok = False
        except ImportError as e:
            logger.info(f"✗ {module_name} (ERROR: {e})")
            all_ok = False
    
    return all_ok

def check_dependencies():
    """Check if all required dependencies are installed."""
    logger.info("\n" + "="*60)
    logger.info("DEPENDENCY CHECK")
    logger.info("="*60)
    
    dependencies = [
        ("chromadb", "0.4.0"),
        ("requests", "2.31.0"),
        ("tiktoken", "0.5.0"),
        ("numpy", "1.24.0"),
    ]
    
    all_ok = True
    for package, min_version in dependencies:
        try:
            mod = __import__(package)
            version = getattr(mod, "__version__", "unknown")
            logger.info(f"✓ {package} ({version}) [required: {min_version}+]")
        except ImportError:
            logger.info(f"✗ {package} (NOT INSTALLED)")
            all_ok = False
    
    return all_ok

def check_syntax():
    """Verify Python syntax in all modules."""
    logger.info("\n" + "="*60)
    logger.info("SYNTAX CHECK")
    logger.info("="*60)
    
    import py_compile
    
    python_files = [
        "app.py",
        "rag/__init__.py",
        "rag/ingest.py",
        "rag/embeddings.py",
        "rag/vectorstore.py",
        "rag/retriever.py",
        "example.py",
    ]
    
    all_ok = True
    for filename in python_files:
        try:
            py_compile.compile(filename, doraise=True)
            logger.info(f"✓ {filename}")
        except py_compile.PyCompileError as e:
            logger.info(f"✗ {filename}: {e}")
            all_ok = False
    
    return all_ok

def check_configuration():
    """Verify configuration files."""
    logger.info("\n" + "="*60)
    logger.info("CONFIGURATION CHECK")
    logger.info("="*60)
    
    # Check .env.example
    env_example = Path(".env.example")
    if env_example.exists():
        logger.info(f"✓ .env.example exists")
        with open(env_example) as f:
            lines = [l for l in f if l.strip() and not l.startswith("#")]
            logger.info(f"  Contains {len(lines)} configuration variables")
    else:
        logger.info(f"✗ .env.example not found")
        return False
    
    # Check requirements.txt
    req_file = Path("requirements.txt")
    if req_file.exists():
        with open(req_file) as f:
            packages = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        logger.info(f"✓ requirements.txt ({len(packages)} packages)")
        for pkg in packages[:3]:
            logger.info(f"    - {pkg}")
        if len(packages) > 3:
            logger.info(f"    ... and {len(packages) - 3} more")
    else:
        logger.info(f"✗ requirements.txt not found")
        return False
    
    return True

def print_summary(results):
    """Print final summary."""
    logger.info("\n" + "="*60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*60)
    
    checks = [
        ("File Structure", results[0]),
        ("Module Imports", results[1]),
        ("Dependencies", results[2]),
        ("Python Syntax", results[3]),
        ("Configuration", results[4]),
    ]
    
    passed = sum(1 for _, result in checks if result)
    total = len(checks)
    
    for check_name, result in checks:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status} - {check_name}")
    
    logger.info("\n" + "="*60)
    if passed == total:
        logger.info(f"✓ ALL CHECKS PASSED ({passed}/{total})")
        logger.info("\nPhase 1 is ready! Next steps:")
        logger.info("  1. Start Ollama: ollama serve")
        logger.info("  2. Pull model: ollama pull nomic-embed-text")
        logger.info("  3. Ingest data: python app.py ingest ./data ./chroma_db")
        logger.info("  4. Query: python app.py query ./chroma_db 'your query' -k 5")
        logger.info("="*60)
        return 0
    else:
        logger.info(f"✗ SOME CHECKS FAILED ({total - passed} issues)")
        logger.info("\nFix the issues above and run this script again.")
        logger.info("="*60)
        return 1

def main():
    try:
        logger.info("\n╔════════════════════════════════════════════════════════╗")
        logger.info("║  BIBLIOPHILE RAG - PHASE 1 VERIFICATION SUITE         ║")
        logger.info("╚════════════════════════════════════════════════════════╝")
        
        results = [
            check_files(),
            check_imports(),
            check_dependencies(),
            check_syntax(),
            check_configuration(),
        ]
        
        return print_summary(results)
    
    except Exception as e:
        logger.error(f"\nFatal error: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())

