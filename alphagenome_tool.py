"""
AlphaGenome Coordinate-Based Prediction Tool
=============================================

Simplified interface for making AlphaGenome predictions using genomic coordinate ranges.
Optimized for batch processing multiple regions for a given tissue and assay.

Features:
- Single or multiple coordinate range input
- Automatic validation of tissue/assay combinations against live API metadata
- Coordinate parsing from strings or BED files
- Batch processing with progress tracking
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Union, Set, Tuple
from dataclasses import dataclass
from pathlib import Path
import os
from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import AlphaGenome
try:
    from alphagenome.data import genome
    from alphagenome.models import dna_client
    ALPHAGENOME_AVAILABLE = True
except ImportError:
    ALPHAGENOME_AVAILABLE = False
    logger.warning("AlphaGenome not installed. Install with: pip install alphagenome")


# ============================================================================
# CONFIGURATION
# ============================================================================

SUPPORTED_LENGTHS = {
    "2KB": 2048,
    "16KB": 16384,
    "100KB": 131072,
    "500KB": 524288,
    "1MB": 1048576,
}

SUPPORTED_LENGTHS_LIST = [2048, 16384, 131072, 524288, 1048576]

ASSAY_ALIASES = {
    "atac": "ATAC",
    "accessibility": "ATAC",
    "dnase": "DNASE",
    "dna_accessibility": "DNASE",
    "rna": "RNA_SEQ",
    "rna_seq": "RNA_SEQ",
    "expression": "RNA_SEQ",
    "cage": "CAGE",
    "tss": "CAGE",
    "procap": "PROCAP",
    "transcription_initiation": "PROCAP",
    "histone": "CHIP_HISTONE",
    "chip_histone": "CHIP_HISTONE",
    "h3k4me3": "CHIP_HISTONE",
    "h3k27ac": "CHIP_HISTONE",
    "h3k27me3": "CHIP_HISTONE",
    "tf": "CHIP_TF",
    "chip_tf": "CHIP_TF",
    "transcription_factor": "CHIP_TF",
    "ctcf": "CHIP_TF",
    "splice": "SPLICE_SITES",
    "splice_sites": "SPLICE_SITES",
    "splicing": "SPLICE_SITES",
    "splice_usage": "SPLICE_SITE_USAGE",
    "splice_site_usage": "SPLICE_SITE_USAGE",
    "junctions": "SPLICE_JUNCTIONS",
    "splice_junctions": "SPLICE_JUNCTIONS",
    "contacts": "CONTACT_MAPS",
    "contact_maps": "CONTACT_MAPS",
    "hic": "CONTACT_MAPS",
    "3d": "CONTACT_MAPS",
}


@dataclass
class CoordinateRange:
    """Represents a genomic coordinate range (0-based, half-open)."""
    chrom: str
    start: int
    end: int
    name: Optional[str] = None
    
    def __post_init__(self):
        if not self.chrom.startswith('chr'):
            self.chrom = f"chr{self.chrom}"
        if self.start < 0:
            raise ValueError(f"Start position must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise ValueError(f"End must be > start, got {self.end} <= {self.start}")
    
    @property
    def width(self) -> int:
        return self.end - self.start
    
    def to_interval(self) -> Any:
        return genome.Interval(
            chromosome=self.chrom,
            start=self.start,
            end=self.end
        )
    
    def __str__(self) -> str:
        name_str = f" ({self.name})" if self.name else ""
        return f"{self.chrom}:{self.start:,}-{self.end:,}{name_str}"
    
    @classmethod
    def from_string(cls, coord_str: str, name: Optional[str] = None) -> "CoordinateRange":
        clean = coord_str.replace(" ", "").replace(",", "")
        if ":" not in clean:
            raise ValueError(f"Invalid format: {coord_str}. Expected 'chr:start-end'")
        chrom, rest = clean.split(":", 1)
        if "-" not in rest:
            pos = int(rest)
            return cls(chrom, pos, pos + 1, name)
        start_str, end_str = rest.split("-", 1)
        return cls(chrom, int(start_str), int(end_str), name)
    
    @classmethod
    def from_bed_line(cls, line: str) -> "CoordinateRange":
        parts = line.strip().split("\t")
        chrom = parts[0]
        start = int(parts[1])
        end = int(parts[2])
        name = parts[3] if len(parts) > 3 else None
        return cls(chrom, start, end, name)


# ============================================================================
# ONTOLOGY VALIDATOR
# ============================================================================

class OntologyValidator:
    """Validates tissue/assay combinations against live API metadata."""
    
    def __init__(self, client: Any):
        self.client = client
        self._metadata_cache = {}
        self._ontology_cache = {}
    
    def get_available_ontologies(self, assay_type: str) -> Set[str]:
        """
        Fetch available ontology CURIEs for a given assay type from API metadata.
        
        Args:
            assay_type: Official assay type name (e.g., 'ATAC', 'CHIP_TF')
            
        Returns:
            Set of available ontology CURIE strings
        """
        # Check cache first
        if assay_type in self._ontology_cache:
            return self._ontology_cache[assay_type]
        
        # Fetch metadata from API
        try:
            metadata = self.client.output_metadata(organism=dna_client.Organism.HOMO_SAPIENS)
            self._metadata_cache['full'] = metadata
        except Exception as e:
            logger.error(f"Failed to fetch metadata: {e}")
            return set()
        
        # Extract ontology_curie column for the specific assay
        assay_key = assay_type.lower()
        if not hasattr(metadata, assay_key):
            logger.warning(f"Metadata not found for assay: {assay_type}")
            return set()
        
        assay_metadata = getattr(metadata, assay_key)
        
        # Extract unique ontology_curie values
        if 'ontology_curie' in assay_metadata.columns:
            ontologies = set(assay_metadata['ontology_curie'].dropna().unique())
        elif 'ontology_id' in assay_metadata.columns:
            ontologies = set(assay_metadata['ontology_id'].dropna().unique())
        else:
            # Fallback: look for any column containing 'ontology'
            ontology_cols = [c for c in assay_metadata.columns if 'ontology' in c.lower()]
            if ontology_cols:
                ontologies = set(assay_metadata[ontology_cols[0]].dropna().unique())
            else:
                ontologies = set()
                logger.warning(f"No ontology column found in metadata for {assay_type}")
        
        self._ontology_cache[assay_type] = ontologies
        logger.info(f"Found {len(ontologies)} valid ontologies for {assay_type}")
        return ontologies
    
    def get_biosample_summary(self, assay_type: str) -> pd.DataFrame:
        """Get summary of available biosamples for an assay type."""
        metadata = self.client.output_metadata(organism=dna_client.Organism.HOMO_SAPIENS)
        assay_key = assay_type.lower()
        
        if not hasattr(metadata, assay_key):
            return pd.DataFrame()
        
        assay_metadata = getattr(metadata, assay_key)
        
        # Summarize by biosample/ontology
        summary_cols = ['biosample_name', 'ontology_curie', 'ontology_name']
        available_cols = [c for c in summary_cols if c in assay_metadata.columns]
        
        if available_cols:
            return assay_metadata[available_cols].drop_duplicates().sort_values(by=available_cols[0])
        return assay_metadata.head()
    
    def validate_ontology(self, ontology_id: str, assay_type: str) -> Tuple[bool, str]:
        """
        Validate if an ontology ID is available for a specific assay.
        
        Args:
            ontology_id: Ontology CURIE (e.g., 'UBERON:0002048') or name
            assay_type: Assay type to check against
            
        Returns:
            (is_valid, message) tuple
        """
        available = self.get_available_ontologies(assay_type)
        
        if not available:
            return False, f"No metadata available for assay {assay_type}"
        
        # Direct match
        if ontology_id in available:
            return True, f"Valid: {ontology_id} is available for {assay_type}"
        
        # Try to resolve if it's a name rather than CURIE
        metadata = self._metadata_cache.get('full')
        assay_key = assay_type.lower()
        
        if metadata and hasattr(metadata, assay_key):
            assay_meta = getattr(metadata, assay_key)
            
            # Search in ontology_name or biosample_name columns
            name_cols = [c for c in assay_meta.columns if 'name' in c.lower()]
            for col in name_cols:
                matches = assay_meta[assay_meta[col].str.lower() == ontology_id.lower()]
                if not matches.empty:
                    correct_curie = matches.iloc[0].get('ontology_curie', ontology_id)
                    return False, f"Invalid CURIE format. Did you mean: {correct_curie} ({ontology_id})?"
        
        # Suggest similar ontologies
        suggestions = [o for o in available if ontology_id.split(':')[0] in o]
        if suggestions:
            return False, f"Invalid ontology: {ontology_id}. Similar available: {suggestions[:5]}"
        
        return False, f"Invalid ontology: {ontology_id}. Not found in {len(available)} available ontologies for {assay_type}"
    
    def find_ontology_by_name(self, name: str, assay_type: str) -> Optional[str]:
        """Find ontology CURIE by biosample/ontology name."""
        metadata = self.client.output_metadata(organism=dna_client.Organism.HOMO_SAPIENS)
        assay_key = assay_type.lower()
        
        if not hasattr(metadata, assay_key):
            return None
        
        assay_meta = getattr(metadata, assay_key)
        name_lower = name.lower()
        
        # Search across relevant columns
        search_cols = [c for c in assay_meta.columns if any(x in c.lower() for x in ['name', 'biosample', 'tissue'])]
        
        for col in search_cols:
            matches = assay_meta[assay_meta[col].str.lower().str.contains(name_lower, na=False)]
            if not matches.empty:
                return matches.iloc[0].get('ontology_curie', matches.iloc[0].get('ontology_id'))
        
        return None


# ============================================================================
# MAIN PREDICTION CLASS
# ============================================================================

class AlphaGenomePredictor:
    """Streamlined AlphaGenome predictor with ontology validation."""
    
    def __init__(self, api_key: Optional[str] = None, skip_validation: bool = False):
        """
        Initialize predictor.
        
        Args:
            api_key: AlphaGenome API key (or set ALPHAGENOME_API_KEY env var)
            skip_validation: Skip ontology validation (faster but less safe)
        """
        if not ALPHAGENOME_AVAILABLE:
            raise ImportError("AlphaGenome not installed. Run: pip install alphagenome")
        
        self.api_key = api_key or os.getenv("ALPHAGENOME_API_KEY")
        if not self.api_key:
            raise ValueError("API key required. Set ALPHAGENOME_API_KEY or pass to constructor.")
        
        self.client = dna_client.create(self.api_key)
        self.skip_validation = skip_validation
        self.validator = OntologyValidator(self.client) if not skip_validation else None
        
        logger.info("AlphaGenome client initialized")
    
    def _resolve_ontology(self, tissue: str, assay_types: List[str]) -> str:
        """
        Convert tissue name to ontology ID and validate against assays.
        
        Args:
            tissue: Tissue name or ontology CURIE
            assay_types: List of assay types to validate against
            
        Returns:
            Validated ontology CURIE string
            
        Raises:
            ValueError: If tissue cannot be resolved or validated
        """
        # If already a CURIE, validate it
        if ':' in tissue:
            ontology_id = tissue
        else:
            # Try to find by name for each assay type
            found_curie = None
            for assay in assay_types:
                found_curie = self.validator.find_ontology_by_name(tissue, assay)
                if found_curie:
                    break
            
            if found_curie:
                ontology_id = found_curie
                logger.info(f"Resolved '{tissue}' to '{ontology_id}'")
            else:
                ontology_id = tissue  # Pass through, will fail validation
        
        # Validate against all requested assays
        if not self.skip_validation and self.validator:
            validation_errors = []
            for assay in assay_types:
                is_valid, msg = self.validator.validate_ontology(ontology_id, assay)
                if not is_valid:
                    validation_errors.append(f"{assay}: {msg}")
            
            if validation_errors:
                error_msg = f"Ontology validation failed for '{ontology_id}':\n" + "\n".join(validation_errors)
                
                # Try to provide helpful suggestions
                suggestions = []
                for assay in assay_types:
                    available = self.validator.get_available_ontologies(assay)
                    # Find partial matches
                    for ont in available:
                        if tissue.lower() in ont.lower() or any(part in ont.lower() for part in tissue.lower().split()):
                            suggestions.append((assay, ont))
                
                if suggestions:
                    error_msg += f"\n\nDid you mean one of these?\n"
                    for assay, ont in suggestions[:10]:
                        error_msg += f"  - {ont} (for {assay})\n"
                
                raise ValueError(error_msg)
        
        return ontology_id
    
    def _resolve_assays(self, assays: List[str]) -> set:
        """Convert assay names to OutputType enums."""
        output_types = set()
        
        for assay in assays:
            assay_lower = assay.lower().strip()
            official_name = ASSAY_ALIASES.get(assay_lower, assay.upper())
            
            if hasattr(dna_client.OutputType, official_name):
                output_types.add(getattr(dna_client.OutputType, official_name))
            else:
                logger.warning(f"Unknown assay: {assay}")
        
        if not output_types:
            raise ValueError(f"No valid assays. Available: {list(ASSAY_ALIASES.keys())}")
        
        return output_types
    
    def _get_best_sequence_length(self, query_width: int) -> int:
        """Select best sequence length for query."""
        min_required = int(query_width * 2)
        for length in SUPPORTED_LENGTHS_LIST:
            if length >= min_required:
                return length
        return max(SUPPORTED_LENGTHS_LIST)
    
    def get_available_assays(self) -> Dict[str, List[str]]:
        """Get list of available assays and their valid ontologies."""
        if not self.validator:
            return {}
        
        assays = ['ATAC', 'DNASE', 'RNA_SEQ', 'CAGE', 'PROCAP', 
                  'CHIP_HISTONE', 'CHIP_TF', 'SPLICE_SITES', 
                  'SPLICE_SITE_USAGE', 'SPLICE_JUNCTIONS', 'CONTACT_MAPS']
        
        result = {}
        for assay in assays:
            try:
                ontologies = self.validator.get_available_ontologies(assay)
                result[assay] = sorted(list(ontologies))[:20]  # Limit output
            except Exception as e:
                result[assay] = [f"Error: {str(e)}"]
        
        return result
    
    def predict_coordinates(
        self,
        coordinates: Union[CoordinateRange, str, List[Union[CoordinateRange, str]]],
        tissue: str,
        assays: List[str] = None,
        sequence_length: Optional[Union[str, int]] = None,
        aggregate_by: Optional[str] = "mean",
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Predict functional signals for one or more coordinate ranges.
        
        Args:
            coordinates: Single coordinate or list (strings or CoordinateRange objects)
            tissue: Tissue name or ontology CURIE
            assays: List of assays to predict (default: ["atac"])
            sequence_length: "auto", specific size, or None for auto
            aggregate_by: Aggregation method ("mean", "max", "sum", None)
            validate: Whether to validate ontology against assay metadata
        
        Returns:
            Dictionary with predictions or validation errors
        """
        # Normalize inputs
        if isinstance(coordinates, (str, CoordinateRange)):
            coordinates = [coordinates]
        
        coord_ranges = []
        for c in coordinates:
            if isinstance(c, str):
                coord_ranges.append(CoordinateRange.from_string(c))
            else:
                coord_ranges.append(c)
        
        # Resolve assays first (needed for ontology validation)
        if assays is None:
            assays = ["atac"]
        output_types = self._resolve_assays(assays)
        assay_names = [ot.name for ot in output_types]
        
        # Resolve and validate ontology
        try:
            ontology_id = self._resolve_ontology(tissue, assay_names)
        except ValueError as e:
            return {
                "error": str(e),
                "tissue": tissue,
                "assays": assays,
                "coordinates": [str(c) for c in coord_ranges],
                "validation_failed": True
            }
        
        # Process predictions
        results = []
        for coord in tqdm(coord_ranges, desc="Predicting regions"):
            try:
                result = self._predict_single_range(
                    coord, ontology_id, output_types, sequence_length, aggregate_by
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to predict {coord}: {e}")
                results.append({
                    "coordinates": str(coord),
                    "error": str(e)
                })
        
        return {
            "tissue": tissue,
            "ontology_id": ontology_id,
            "assays": assays,
            "validation_skipped": self.skip_validation,
            "results": results
        }
    
    def _predict_single_range(
            self,
            coord: CoordinateRange,
            ontology_id: str,
            output_types: set,
            sequence_length: Optional[Union[str, int]],
            aggregate_by: Optional[str]
        ) -> Dict[str, Any]:
            """Predict for a single coordinate range."""
            
            # 1. Determine Sequence Length
            if sequence_length is None or sequence_length == "auto":
                seq_len = self._get_best_sequence_length(coord.width)
            elif isinstance(sequence_length, str):
                seq_len = SUPPORTED_LENGTHS.get(sequence_length.upper(), 1048576)
            else:
                seq_len = sequence_length

            # 2. Resize Interval (Official Pattern)
            # The SDK requires the query to be resized to a supported model input length
            query_interval = coord.to_interval()
            input_interval = query_interval.resize(seq_len)
            
            # 3. Call API
            output = self.client.predict_interval(
                interval=input_interval,
                requested_outputs=output_types,
                ontology_terms=[ontology_id],
            )
            
            assay_results = {}
            
            for output_type in output_types:
                type_name = output_type.name.lower()
                if not hasattr(output, type_name):
                    continue
                
                track_data_obj = getattr(output, type_name)
                
                # --- REFINED SECTION: Use SDK Helper instead of manual math ---
                try:
                    # This handles resolution (1bp vs 128bp) and stranding automatically
                    # match_resolution=True ensures we don't crash on 128bp tracks
                    sliced_track = track_data_obj.slice_by_interval(
                        query_interval, 
                        match_resolution=True
                    )
                    query_values = sliced_track.values
                    resolution = getattr(track_data_obj, 'resolution', 1) # Just for reporting
                    
                except Exception as e:
                    # Fallback if slice_by_interval fails or API changes
                    logger.warning(f"Slicing failed for {type_name}, using raw window: {e}")
                    query_values = track_data_obj.values
                    resolution = getattr(track_data_obj, 'resolution', 1)

                if query_values.size > 0:
                    real_mean = float(np.nanmean(query_values))
                    real_max = float(np.nanmax(query_values))
                else:
                    real_mean = 0.0
                    real_max = 0.0

                # 5. Handle Aggregation (if output needs to be reduced)
                if aggregate_by and query_values.ndim >= 1 and query_values.size > 0:
                    if aggregate_by == "mean":
                        aggregated = np.nanmean(query_values, axis=0)
                    elif aggregate_by == "max":
                        aggregated = np.nanmax(query_values, axis=0)
                    elif aggregate_by == "sum":
                        aggregated = np.nansum(query_values, axis=0)
                    else:
                        aggregated = query_values
                else:
                    aggregated = query_values

                assay_results[type_name] = {
                    "shape": query_values.shape,
                    "resolution_bp": resolution,
                    # FIX: Use the 'aggregated' var only for the main signal report
                    "aggregate_signal": float(np.nanmean(aggregated)) if np.any(~np.isnan(aggregated)) else 0.0,
                    # FIX: Use 'real_max' derived from raw data for the max peak
                    "max_signal": real_max, 
                }
           
            return {
                "coordinates": str(coord),
                "input_interval": f"{input_interval.chromosome}:{input_interval.start:,}-{input_interval.end:,}",
                "sequence_length": seq_len,
                "assays": assay_results
            }
   
    def predict_from_bed(
        self,
        bed_file: Union[str, Path],
        tissue: str,
        assays: List[str] = None,
        sequence_length: Optional[str] = None,
        aggregate_by: str = "mean",
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Predict for all regions in a BED file."""
        bed_path = Path(bed_file)
        if not bed_path.exists():
            raise FileNotFoundError(f"BED file not found: {bed_file}")
        
        coordinates = []
        with open(bed_path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                if line.strip() and not line.startswith("#"):
                    try:
                        coord = CoordinateRange.from_bed_line(line)
                        coordinates.append(coord)
                    except Exception as e:
                        logger.warning(f"Skipping line {i+1}: {e}")
        
        logger.info(f"Loaded {len(coordinates)} regions from {bed_file}")
        
        return self.predict_coordinates(
            coordinates=coordinates,
            tissue=tissue,
            assays=assays,
            sequence_length=sequence_length,
            aggregate_by=aggregate_by
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_results(results: Dict[str, Any]) -> str:
    """Format prediction results as readable text."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"AlphaGenome Predictions")
    
    if "error" in results:
        lines.append(f"❌ ERROR: {results['error']}")
        if results.get('validation_failed'):
            lines.append("\nOntology validation failed. Available options depend on the assay type.")
            lines.append("Use predictor.get_available_assays() to see valid combinations.")
        lines.append("=" * 70)
        return "\n".join(lines)
    
    lines.append(f"Tissue: {results.get('tissue', 'N/A')} ({results.get('ontology_id', 'N/A')})")
    lines.append(f"Assays: {', '.join(results.get('assays', []))}")
    if results.get('validation_skipped'):
        lines.append("⚠️  Validation skipped")
    lines.append("=" * 70)
    
    for result in results.get('results', []):
        lines.append("")
        if "error" in result:
            lines.append(f"❌ {result['coordinates']}: ERROR - {result['error']}")
            continue
        
        lines.append(f"📍 {result['coordinates']}")
        lines.append(f"   Input window: {result['input_interval']} ({result['sequence_length']:,} bp)")
        
        for assay_name, assay_data in result.get('assays', {}).items():
            lines.append(f"\n   📊 {assay_name.upper()}:")
            lines.append(f"      Resolution: {assay_data['resolution_bp']} bp")
            lines.append(f"      Signal (mean/max): {assay_data['aggregate_signal']:.4f} / {assay_data['max_signal']:.4f}")
    
    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("AlphaGenome Coordinate Prediction Tool with Ontology Validation")
    print("=" * 70)
    
    try:
        predictor = AlphaGenomePredictor()
        
        # Example: Check available assays first
        print("\nAvailable assays and sample ontologies:")
        available = predictor.get_available_assays()
        for assay, ontologies in available.items():
            print(f"  {assay}: {len(ontologies)} ontologies")
            if ontologies and not ontologies[0].startswith("Error"):
                print(f"    Examples: {', '.join(ontologies[:3])}")
        
        # Example: Predict with validation
        print("\n" + "=" * 70)
        print("Example prediction with validation:")
        result = predictor.predict_coordinates(
            coordinates="chr17:7676000-7677000",
            tissue="lung",  # Will be resolved to UBERON:0002048 if valid for ATAC
            assays=["atac"],
            sequence_length="auto"
        )
        print(format_results(result))
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nSetup:")
        print("1. pip install alphagenome")
        print("2. export ALPHAGENOME_API_KEY='your-key'")
