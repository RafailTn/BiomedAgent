#!/usr/bin/env python3
"""
PrimeKG FAST Loader (Polars Edition)
====================================
Uses Polars for fast CSV processing and Kuzu COPY FROM for bulk loading.

Speed comparison:
- Pandas + Row-by-row: 10-15 minutes
- Polars + COPY FROM:  < 1 minute

Usage:
    python load_primekg.py                  # Full load
    python load_primekg.py --limit 100000   # Quick test
    python load_primekg.py --force          # Force reload
"""

import sys
import time
import argparse
import shutil
import logging
from pathlib import Path

# Try importing polars
try:
    import polars as pl
except ImportError:
    sys.exit("Error: Polars is missing. Install it with: pip install polars")

import kuzu

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
    Prepare CSVs using Polars.
    Uses pipe '|' separator to avoid collision with commas in chemical names.
    """
    entities_csv = PRIMEKG_DATA_DIR / "kuzu_entities.csv"
    relations_csv = PRIMEKG_DATA_DIR / "kuzu_relations.csv"
    
    logger.info("Reading PrimeKG data with Polars...")
    
    # === FIX: FORCE IDs TO BE STRINGS ===
    # We use schema_overrides to prevent Polars from guessing 'x_id' is an integer
    df = pl.read_csv(
        edges_file, 
        infer_schema_length=10000,
        schema_overrides={
            "x_id": pl.Utf8, 
            "y_id": pl.Utf8,
            "x_type": pl.Utf8,
            "y_type": pl.Utf8
        }
    )
    
    if limit:
        df = df.head(limit)
        logger.info(f"Limited to {limit:,} edges")
    
    total_edges = len(df)
    logger.info(f"Total edges: {total_edges:,}")
    
    # === Extract unique entities ===
    logger.info("Extracting unique entities...")
    
    # Select x and y columns, rename them to standard names
    subjects = df.select([
        pl.col('x_id').alias('orig_id'),
        pl.col('x_type').alias('type'),
        pl.col('x_name').alias('name')
    ])
    
    objects = df.select([
        pl.col('y_id').alias('orig_id'),
        pl.col('y_type').alias('type'),
        pl.col('y_name').alias('name')
    ])
    
    # Concat and get unique entities by ID
    all_entities = pl.concat([subjects, objects]).unique(subset=['orig_id'])
    
    logger.info(f"Found {len(all_entities):,} unique entities")
    
    # === Create Entity CSV ===
    logger.info("Creating entities CSV for Kuzu...")
    
    # Transform to Kuzu schema using Polars expressions
    # Note: we don't need .cast(pl.Utf8) for IDs anymore since we forced it at read time
    entity_df = all_entities.select([
        (pl.lit('primekg:') + pl.col('orig_id')).alias('id'),
        pl.col('name').fill_null("").cast(pl.Utf8).alias('name'),
        pl.col('type').fill_null("").cast(pl.Utf8).alias('type'),
        pl.lit("").alias('cui'),
        pl.lit("primekg").alias('source'),
        pl.lit("").alias('description')
    ])
    
    # Write with PIPE separator
    entity_df.write_csv(entities_csv, separator='|')
    logger.info(f"  Saved: {entities_csv} ({len(entity_df):,} rows)")
    
    # === Create Relations CSV ===
    logger.info("Creating relations CSV for Kuzu...")
    
    rel_df = df.select([
        (pl.lit('primekg:') + pl.col('x_id')).alias('from'),
        (pl.lit('primekg:') + pl.col('y_id')).alias('to'),
        pl.col('relation').fill_null("").cast(pl.Utf8).alias('relation'),
        pl.lit("primekg").alias('source'),
        pl.lit(1.0).alias('confidence'),
        pl.lit("").alias('pmid')
    ])
    
    # Write with PIPE separator
    rel_df.write_csv(relations_csv, separator='|')
    logger.info(f"  Saved: {relations_csv} ({len(rel_df):,} rows)")
    
    return entities_csv, relations_csv

def bulk_load_into_kuzu(entities_csv: Path, relations_csv: Path, force: bool = False):
    """
    Load CSVs into Kuzu using COPY FROM.
    Refined to handle 'Zombie Files' (where DB exists as a file instead of a folder).
    """
    # 1. CHECK EXISTING DB
    if KUZU_DB_PATH.exists() and not force:
        try:
            # Try to connect regardless of whether it is a file or folder
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
            logger.warning(f"Existing DB check failed (will reload): {e}")

    # 2. CLEANUP: Delete existing DB whether it is a file OR a folder
    if KUZU_DB_PATH.exists():
        logger.info("Removing existing database...")
        try:
            if KUZU_DB_PATH.is_dir():
                shutil.rmtree(KUZU_DB_PATH)
            else:
                # This fixes the "NotADirectoryError" if it was created as a file
                KUZU_DB_PATH.unlink()
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")
    
    # 3. CREATE NEW DATABASE
    logger.info(f"Creating Kuzu database: {KUZU_DB_PATH}")
    
    # We let Kuzu handle the creation logic (file vs folder) automatically
    # by passing the path string directly.
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
        abs_path = str(entities_csv.resolve())
        # Note: Using pipe delimiter '|' to handle chemical names with commas
        conn.execute(f"COPY Entity FROM '{abs_path}' (header=true, delimiter='|')")
        
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
        conn.execute(f"COPY RELATES_TO FROM '{abs_path}' (header=true, delimiter='|')")
        
        result = conn.execute("MATCH ()-[r:RELATES_TO]->() RETURN count(r)")
        rel_count = result.get_next()[0]
        logger.info(f"  ✓ {rel_count:,} relations loaded in {time.time()-start:.1f}s")
        
    except Exception as e:
        logger.error(f"COPY RELATES_TO failed: {e}")
        raise
    
    return entity_count, rel_count

def main():
    parser = argparse.ArgumentParser(description='Fast PrimeKG loader using Polars and Kuzu')
    parser.add_argument('--limit', type=int, help='Limit edges (e.g., 100000 for testing)')
    parser.add_argument('--force', action='store_true', help='Force reload even if exists')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("PrimeKG FAST Loader (Polars Edition)")
    print("="*60)
    print(f"Database: {KUZU_DB_PATH}")
    
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
        
    except Exception as e:
        logger.error(f"Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
