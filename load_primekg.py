#!/usr/bin/env python3
"""
PrimeKG Loader - Standalone Script
==================================
Run this ONCE before using the main research agent.

Usage:
    python load_primekg.py                  # Full load (10-15 min)
    python load_primekg.py --limit 100000   # Quick test (~2 min)
    python load_primekg.py --force          # Force reload

This script loads PrimeKG into Kuzu WITHOUT loading any GPU models,
so it won't cause CUDA OOM errors.
"""

import sys
import argparse
from pathlib import Path

# Add script directory to path
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

from knowledge_graph import KuzuGraphDB, PrimeKGLoader

# Paths - same as main agent
KUZU_DB_PATH = SCRIPT_DIR / "kuzu_biomedical_kg"
PRIMEKG_DATA_DIR = SCRIPT_DIR / "primekg_data"


def main():
    parser = argparse.ArgumentParser(description='Load PrimeKG into Kuzu (CPU only, no GPU)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of edges (e.g., 100000 for quick test)')
    parser.add_argument('--force', action='store_true',
                       help='Force reload even if already loaded')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("PrimeKG Loader (CPU Only - No GPU Required)")
    print("="*60)
    print(f"Database: {KUZU_DB_PATH}")
    print(f"Data dir: {PRIMEKG_DATA_DIR}")
    if args.limit:
        print(f"Limit: {args.limit:,} edges")
    print("="*60 + "\n")
    
    # Initialize database
    kg_db = KuzuGraphDB(KUZU_DB_PATH)
    kg_db.initialize()
    
    # Check if already loaded
    if not args.force:
        stats = kg_db.get_stats()
        if stats.get('entities', 0) > 50000:
            print(f"✓ PrimeKG already loaded!")
            print(f"  Entities: {stats['entities']:,}")
            print(f"  Relations: {stats['relations']:,}")
            print("\nUse --force to reload.")
            return
    
    # Load PrimeKG
    loader = PrimeKGLoader(PRIMEKG_DATA_DIR)
    
    if args.force:
        # Remove existing data for fresh load
        import shutil
        if KUZU_DB_PATH.exists():
            print("Removing existing database for fresh load...")
            kg_db.close()
            shutil.rmtree(KUZU_DB_PATH)
            kg_db = KuzuGraphDB(KUZU_DB_PATH)
            kg_db.initialize()
    
    relation_count = loader.load_into_kuzu(kg_db, limit=args.limit)
    
    # Final stats
    stats = kg_db.get_stats()
    print("\n" + "="*60)
    print("COMPLETE!")
    print(f"  Entities: {stats['entities']:,}")
    print(f"  Relations: {stats['relations']:,}")
    print("="*60)
    print("\nThe main research agent will now detect this database")
    print("and skip the loading step automatically.")
    print()
    
    kg_db.close()


if __name__ == "__main__":
    main()
