#!/usr/bin/env python3
"""
PrimeKG FAST Loader
===================
Uses Kuzu COPY FROM for bulk loading - loads 4M relations in ~1-2 minutes!

Speed comparison:
- Row-by-row: 10-15 minutes (or hours)
- COPY FROM:  1-2 minutes

Usage:
    python load_primekg_fast.py                  # Full load (~2 min)
    python load_primekg_fast.py --limit 100000   # Quick test (~20 sec)
    python load_primekg_fast.py --force          # Force reload
"""

import sys
import time
import argparse
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.resolve()
KUZU_DB_PATH = SCRIPT_DIR / "kuzu_biomedical_kg"
PRIMEKG_DATA_DIR = SCRIPT_DIR / "primekg_data"
PRIMEKG_URL = "https://dataverse.harvard.edu/api/access/datafile/6180620"


def download_primekg():
    """Download PrimeKG if not present."""
    import requests
    
    edges_file = PRIMEKG_DATA_DIR / "kg.csv"
    if edges_file.exists():
        size_mb = edges_file.stat().st_size // (1024 * 1024)
        logger.info(f"PrimeKG CSV exists ({size_mb} MB)")
        return edges_file
    
    PRIMEKG_DATA_DIR.mkdir(exist_ok=True)
    logger.info("Downloading PrimeKG (~300MB)...")
    
    r = requests.get(PRIMEKG_URL, stream=True)
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0))
    
    with open(edges_file, 'wb') as f:
        dl = 0
        for chunk in r.iter_content(65536):
            f.write(chunk)
            dl += len(chunk)
            if total:
                print(f"\r  {dl // (1024*1024)} / {total // (1024*1024)} MB ({100*dl//total}%)", end='')
    print()
    return edges_file


def prepare_csvs_for_kuzu(edges_file: Path, limit: int = None):
    """
    Prepare CSVs in Kuzu's exact format for COPY FROM.
    
    Key requirements:
    - Entity CSV: columns must match table schema order
    - Relation CSV: first two columns must be FROM and TO node IDs
    """
    import pandas as pd
    
    entities_csv = PRIMEKG_DATA_DIR / "kuzu_entities.csv"
    relations_csv = PRIMEKG_DATA_DIR / "kuzu_relations.csv"
    
    logger.info("Reading PrimeKG data...")
    df = pd.read_csv(edges_file, low_memory=False)
    
    if limit:
        df = df.head(limit)
        logger.info(f"Limited to {limit:,} edges")
    
    total_edges = len(df)
    logger.info(f"Total edges: {total_edges:,}")
    
    # === Extract unique entities ===
    logger.info("Extracting unique entities...")
    subjects = df[['x_id', 'x_type', 'x_name']].drop_duplicates()
    objects = df[['y_id', 'y_type', 'y_name']].drop_duplicates()
    
    # Rename and combine
    subjects.columns = ['orig_id', 'type', 'name']
    objects.columns = ['orig_id', 'type', 'name']
    all_entities = pd.concat([subjects, objects]).drop_duplicates(subset=['orig_id'])
    
    logger.info(f"Found {len(all_entities):,} unique entities")
    
    # === Create Entity CSV ===
    # Schema: id STRING, name STRING, type STRING, cui STRING, source STRING, description STRING
    logger.info("Creating entities CSV for Kuzu...")
    
    entity_df = pd.DataFrame({
        'id': 'primekg:' + all_entities['orig_id'].astype(str),
        'name': all_entities['name'].fillna('').astype(str),
        'type': all_entities['type'].fillna('').astype(str),
        'cui': '',
        'source': 'primekg',
        'description': ''
    })
    entity_df.to_csv(entities_csv, index=False)
    logger.info(f"  Saved: {entities_csv} ({len(entity_df):,} rows)")
    
    # === Create Relations CSV ===
    # For REL TABLE COPY: FROM_id, TO_id, prop1, prop2, ...
    # Schema: relation STRING, source STRING, confidence DOUBLE, pmid STRING
    logger.info("Creating relations CSV for Kuzu...")
    
    rel_df = pd.DataFrame({
        'from': 'primekg:' + df['x_id'].astype(str),
        'to': 'primekg:' + df['y_id'].astype(str),
        'relation': df['relation'].fillna('').astype(str),
        'source': 'primekg',
        'confidence': 1.0,
        'pmid': ''
    })
    rel_df.to_csv(relations_csv, index=False)
    logger.info(f"  Saved: {relations_csv} ({len(rel_df):,} rows)")
    
    return entities_csv, relations_csv


def bulk_load_into_kuzu(entities_csv: Path, relations_csv: Path, force: bool = False):
    """
    Load CSVs into Kuzu using COPY FROM (very fast!).
    """
    import kuzu
    
    # Check existing database
    if KUZU_DB_PATH.exists() and not force:
        try:
            db = kuzu.Database(str(KUZU_DB_PATH))
            conn = kuzu.Connection(db)
            result = conn.execute("MATCH (e:Entity) RETURN count(e)")
            entity_count = result.get_next()[0]
            
            if entity_count > 50000:
                result = conn.execute("MATCH ()-[r:RELATES_TO]->() RETURN count(r)")
                rel_count = result.get_next()[0]
                logger.info(f"✓ Already loaded: {entity_count:,} entities, {rel_count:,} relations")
                logger.info("Use --force to reload")
                return entity_count, rel_count
        except Exception as e:
            logger.warning(f"Could not read existing DB: {e}")
    
    # Remove existing for clean COPY
    if KUZU_DB_PATH.exists():
        logger.info("Removing existing database...")
        shutil.rmtree(KUZU_DB_PATH)
    
    # Create fresh database
    logger.info(f"Creating Kuzu database: {KUZU_DB_PATH}")
    db = kuzu.Database(str(KUZU_DB_PATH))
    conn = kuzu.Connection(db)
    
    # Create schema
    logger.info("Creating schema...")
    conn.execute("""
        CREATE NODE TABLE Entity(
            id STRING PRIMARY KEY, name STRING, type STRING,
            cui STRING, source STRING, description STRING)
    """)
    conn.execute("""
        CREATE NODE TABLE Document(
            pmid STRING PRIMARY KEY, title STRING, year STRING, has_full_text BOOLEAN)
    """)
    conn.execute("""
        CREATE REL TABLE MENTIONS(FROM Document TO Entity, count INT32, context STRING)
    """)
    conn.execute("""
        CREATE REL TABLE RELATES_TO(
            FROM Entity TO Entity, relation STRING, source STRING, confidence DOUBLE, pmid STRING)
    """)
    
    # === BULK LOAD ENTITIES ===
    logger.info("Loading entities with COPY FROM...")
    start = time.time()
    
    try:
        # Use absolute path for COPY command
        abs_path = str(entities_csv.resolve())
        conn.execute(f"COPY Entity FROM '{abs_path}' (header=true)")
        
        result = conn.execute("MATCH (e:Entity) RETURN count(e)")
        entity_count = result.get_next()[0]
        logger.info(f"  ✓ {entity_count:,} entities loaded in {time.time()-start:.1f}s")
        
    except Exception as e:
        logger.error(f"COPY Entity failed: {e}")
        raise
    
    # === BULK LOAD RELATIONS ===
    logger.info("Loading relations with COPY FROM...")
    start = time.time()
    
    try:
        abs_path = str(relations_csv.resolve())
        conn.execute(f"COPY RELATES_TO FROM '{abs_path}' (header=true)")
        
        result = conn.execute("MATCH ()-[r:RELATES_TO]->() RETURN count(r)")
        rel_count = result.get_next()[0]
        logger.info(f"  ✓ {rel_count:,} relations loaded in {time.time()-start:.1f}s")
        
    except Exception as e:
        logger.error(f"COPY RELATES_TO failed: {e}")
        raise
    
    return entity_count, rel_count


def main():
    parser = argparse.ArgumentParser(description='Fast PrimeKG loader using Kuzu COPY FROM')
    parser.add_argument('--limit', type=int, help='Limit edges (e.g., 100000 for testing)')
    parser.add_argument('--force', action='store_true', help='Force reload even if exists')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("PrimeKG FAST Loader (Kuzu COPY FROM)")
    print("="*60)
    print(f"Database: {KUZU_DB_PATH}")
    if args.limit:
        print(f"Limit: {args.limit:,} edges")
    print("="*60 + "\n")
    
    total_start = time.time()
    
    try:
        # 1. Download PrimeKG
        edges_file = download_primekg()
        
        # 2. Prepare CSVs
        entities_csv, relations_csv = prepare_csvs_for_kuzu(edges_file, args.limit)
        
        # 3. Bulk load
        entity_count, rel_count = bulk_load_into_kuzu(entities_csv, relations_csv, args.force)
        
        total_time = time.time() - total_start
        
        print("\n" + "="*60)
        print(f"✓ COMPLETE in {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"  Entities:  {entity_count:,}")
        print(f"  Relations: {rel_count:,}")
        print("="*60)
        print("\nThe research agent will detect this database automatically.")
        
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
