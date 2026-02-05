"""
AlphaGenome Tool v3 - Complete Implementation Using Official API
================================================================

This implementation uses the proper AlphaGenome API patterns:
1. Uses `predict_interval()` with `genome.Interval` (not manual sequence fetching)
2. Properly handles multi-track outputs with metadata
3. Includes ALL 11 output types with correct units/interpretation
4. Uses official variant scoring patterns
5. Handles TF-specific and histone-specific tracks correctly

Output Types Supported:
- ATAC, DNASE: Chromatin accessibility (normalized insertion signal, 1bp)
- RNA_SEQ, CAGE, PROCAP: Gene expression (normalized read signal, 1bp)
- CHIP_HISTONE: Histone modifications (fold-change over control, 128bp)
- CHIP_TF: Transcription factor binding (fold-change over control, 128bp)
- SPLICE_SITES: Splice site probability (0-1, 1bp)
- SPLICE_SITE_USAGE: Fraction of transcripts using site (0-1, 1bp)
- SPLICE_JUNCTIONS: Junction read counts (normalized, 1bp)
- CONTACT_MAPS: 3D chromatin contacts (log-fold over distance, 2048bp)
"""

import numpy as np
import logging
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Try to import AlphaGenome
try:
    from alphagenome.data import genome
    from alphagenome.models import dna_client
    from alphagenome.models import variant_scorers
    ALPHAGENOME_AVAILABLE = True
except ImportError:
    ALPHAGENOME_AVAILABLE = False
    logger.warning("AlphaGenome not installed. Install with: pip install alphagenome")
    # Create dummy classes for type hints
    genome = None
    dna_client = None
    variant_scorers = None


# ============================================================================
# OUTPUT TYPE METADATA
# ============================================================================

@dataclass
class OutputTypeInfo:
    """Metadata about each AlphaGenome output type."""
    name: str
    description: str
    units: str
    resolution_bp: int
    interpretation: str
    has_strand: bool = False
    has_tissue: bool = True
    extra_metadata: Optional[List[str]] = None  # e.g., ['transcription_factor', 'histone_mark']


OUTPUT_TYPE_INFO = {
    "ATAC": OutputTypeInfo(
        name="ATAC",
        description="Chromatin accessibility (ATAC-seq)",
        units="Normalized insertion signal (100M insertions)",
        resolution_bp=1,
        interpretation="Higher values = more accessible/open chromatin. Compare across regions or tissues.",
        has_strand=False,
    ),
    "DNASE": OutputTypeInfo(
        name="DNASE", 
        description="Chromatin accessibility (DNase-seq)",
        units="Normalized insertion signal (100M insertions)",
        resolution_bp=1,
        interpretation="Higher values = more accessible/open chromatin. DNase often more sensitive than ATAC.",
        has_strand=False,
    ),
    "RNA_SEQ": OutputTypeInfo(
        name="RNA_SEQ",
        description="Gene expression (RNA-seq)",
        units="Normalized read signal (RPM-like, 100M reads)",
        resolution_bp=1,
        interpretation="Higher values = more transcription. Stranded tracks show direction.",
        has_strand=True,
        extra_metadata=["gtex_tissue"],
    ),
    "CAGE": OutputTypeInfo(
        name="CAGE",
        description="Transcription start site activity (CAGE-seq)",
        units="Normalized read signal",
        resolution_bp=1,
        interpretation="Peaks indicate active transcription start sites. High near promoters.",
        has_strand=True,
    ),
    "PROCAP": OutputTypeInfo(
        name="PROCAP",
        description="Nascent transcription (PRO-cap)",
        units="Normalized read signal",
        resolution_bp=1,
        interpretation="Captures active RNA polymerase. More precise TSS than CAGE.",
        has_strand=True,
    ),
    "CHIP_HISTONE": OutputTypeInfo(
        name="CHIP_HISTONE",
        description="Histone modifications (ChIP-seq)",
        units="Fold-change over control (summed per 128bp bin)",
        resolution_bp=128,
        interpretation="Values >1 = enriched over input control. H3K4me3=promoters, H3K27ac=active enhancers, H3K27me3=repressed. Strong peaks typically 1000-20000. Compare across regions rather than using absolute thresholds.",
        has_strand=False,
        extra_metadata=["histone_mark"],
    ),
    "CHIP_TF": OutputTypeInfo(
        name="CHIP_TF",
        description="Transcription factor binding (ChIP-seq)",
        units="Fold-change over control (summed per 128bp bin)",
        resolution_bp=128,
        interpretation="Values >1 = TF binding enriched. K562/HepG2 have most TFs (269/501). Other tissues mainly CTCF/POLR2A. Strong peaks typically 1000-20000. Compare across regions rather than using absolute thresholds.",
        has_strand=False,
        extra_metadata=["transcription_factor"],
    ),
    "SPLICE_SITES": OutputTypeInfo(
        name="SPLICE_SITES",
        description="Splice site probability",
        units="Probability (0-1)",
        resolution_bp=1,
        interpretation="Higher = more likely to be a splice site. Separate tracks for donor/acceptor and +/- strand.",
        has_strand=True,
        has_tissue=False,  # Tissue-agnostic!
    ),
    "SPLICE_SITE_USAGE": OutputTypeInfo(
        name="SPLICE_SITE_USAGE",
        description="Fraction of transcripts using splice site",
        units="Fraction (0-1)",
        resolution_bp=1,
        interpretation="What fraction of spanning reads use this splice site. 1.0 = constitutive, <1 = alternative.",
        has_strand=True,
    ),
    "SPLICE_JUNCTIONS": OutputTypeInfo(
        name="SPLICE_JUNCTIONS",
        description="Splice junction read counts",
        units="Normalized junction signal",
        resolution_bp=1,
        interpretation="Predicted split-read counts spanning introns. Higher = more splicing events.",
        has_strand=True,
    ),
    "CONTACT_MAPS": OutputTypeInfo(
        name="CONTACT_MAPS",
        description="3D chromatin contacts (Hi-C/Micro-C)",
        units="Log-fold over genomic distance expectation",
        resolution_bp=2048,
        interpretation="Positive = more contact than expected for that distance. Detects TADs, loops, compartments.",
        has_strand=False,
    ),
}


# ============================================================================
# TISSUE/ONTOLOGY MAPPING
# ============================================================================

TISSUE_ONTOLOGY_MAP = {
    "Adipose - Subcutaneous": "UBERON:0002190",
    "Adipose - Visceral (Omentum)": "UBERON:0010414",
    "Adrenal Gland": "UBERON:0002369",
    "Artery - Aorta": "UBERON:0001496",
    "Artery - Coronary": "UBERON:0001621",
    "Artery - Tibial": "UBERON:0007610",
    "Bladder": "UBERON:0001255",
    "Brain - Amygdala": "UBERON:0001876",
    "Brain - Anterior cingulate cortex (BA24)": "UBERON:0009835",
    "Brain - Caudate (basal ganglia)": "UBERON:0001873",
    "Brain - Cerebellar Hemisphere": "UBERON:0002245",
    "Brain - Cerebellum": "UBERON:0002037",
    "Brain - Cortex": "UBERON:0001870",
    "Brain - Frontal Cortex (BA9)": "UBERON:0009834",
    "Brain - Hippocampus": "UBERON:0001954",
    "Brain - Hypothalamus": "UBERON:0001898",
    "Brain - Nucleus accumbens (basal ganglia)": "UBERON:0001882",
    "Brain - Putamen (basal ganglia)": "UBERON:0001874",
    "Brain - Spinal cord (cervical c-1)": "UBERON:0006469",
    "Brain - Substantia nigra": "UBERON:0002038",
    "Breast - Mammary Tissue": "UBERON:0008367",
    "Cells - Cultured fibroblasts": "EFO:0002009",
    "Cells - EBV-transformed lymphocytes": "EFO:0000572",
    "Cervix - Ectocervix": "UBERON:0012249",
    "Cervix - Endocervix": "UBERON:0000458",
    "Colon - Sigmoid": "UBERON:0001159",
    "Colon - Transverse": "UBERON:0001157",
    "Esophagus - Gastroesophageal Junction": "UBERON:0004550",
    "Esophagus - Mucosa": "UBERON:0006920",
    "Esophagus - Muscularis": "UBERON:0004648",
    "Fallopian Tube": "UBERON:0003889",
    "Heart - Atrial Appendage": "UBERON:0006631",
    "Heart - Left Ventricle": "UBERON:0006566",
    "Kidney - Cortex": "UBERON:0001225",
    "Kidney - Medulla": "UBERON:0001293",
    "Liver": "UBERON:0001114",
    "Lung": "UBERON:0008952",
    "Minor Salivary Gland": "UBERON:0006330",
    "Muscle - Skeletal": "UBERON:0011907",
    "Nerve - Tibial": "UBERON:0001323",
    "Ovary": "UBERON:0000992",
    "Pancreas": "UBERON:0001150",
    "Pituitary": "UBERON:0000007",
    "Prostate": "UBERON:0002367",
    "Skin - Not Sun Exposed (Suprapubic)": "UBERON:0036149",
    "Skin - Sun Exposed (Lower leg)": "UBERON:0004264",
    "Small Intestine - Terminal Ileum": "UBERON:0001211",
    "Spleen": "UBERON:0002106",
    "Stomach": "UBERON:0000945",
    "Testis": "UBERON:0000473",
    "Thyroid": "UBERON:0002046",
    "Uterus": "UBERON:0000995",
    "Vagina": "UBERON:0000996",
    "Whole Blood": "UBERON:0013756",
    "K562 (Myeloid Leukemia)": "EFO:0002067",
    "HepG2 (Liver Carcinoma)": "EFO:0001187",
}

# Reverse mapping for display
ONTOLOGY_TO_NAME = {v: k for k, v in TISSUE_ONTOLOGY_MAP.items()}


# ============================================================================
# ASSAY NAME MAPPING  
# ============================================================================

ASSAY_ALIASES = {
    # Short names -> Official OutputType names
    "atac": "ATAC",
    "accessibility": "ATAC",
    "dnase": "DNASE",
    "rna": "RNA_SEQ",
    "rna_seq": "RNA_SEQ",
    "expression": "RNA_SEQ",
    "cage": "CAGE",
    "tss": "CAGE",
    "procap": "PROCAP",
    "histone": "CHIP_HISTONE",
    "chip_histone": "CHIP_HISTONE",
    "h3k4me3": "CHIP_HISTONE",
    "h3k27ac": "CHIP_HISTONE",
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


# ============================================================================
# MAIN HANDLER CLASS
# ============================================================================

class AlphaGenomeHandler:
    """
    Handler for Google DeepMind's AlphaGenome API.
    
    Uses the official API patterns:
    - predict_interval() with genome.Interval objects
    - Proper coordinate tracking via interval.resize()
    - Full track metadata access
    - Built-in variant scoring
    """
    
    # Valid sequence lengths
    SEQUENCE_LENGTHS = {
        "2KB": 2048,
        "16KB": 16384,
        "100KB": 102400,
        "500KB": 524288,
        "1MB": 1048576,
    }
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with API key (or from ALPHAGENOME_API_KEY env var)."""
        self.api_key = api_key or os.getenv("ALPHAGENOME_API_KEY", "")
        self.client = None
        self.output_metadata = None
        
        if not ALPHAGENOME_AVAILABLE:
            logger.error("AlphaGenome library not installed")
            return
            
        if not self.api_key:
            logger.warning("No API key provided. Set ALPHAGENOME_API_KEY environment variable.")
            return
            
        try:
            self.client = dna_client.create(self.api_key)
            # Cache output metadata for track info
            self.output_metadata = self.client.output_metadata(
                organism=dna_client.Organism.HOMO_SAPIENS
            )
            logger.info("AlphaGenome client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize AlphaGenome client: {e}")
    
    def _resolve_tissue(self, tissue: str) -> str:
        """Convert tissue name to ontology ID."""
        tissue_lower = tissue.lower().replace(" ", "_").replace("-", "_")
        
        # Check if already an ontology ID
        if ":" in tissue:
            return tissue
            
        # Look up in map
        if tissue_lower in TISSUE_ONTOLOGY_MAP:
            return TISSUE_ONTOLOGY_MAP[tissue_lower]
        
        # Try partial match
        for key, value in TISSUE_ONTOLOGY_MAP.items():
            if tissue_lower in key or key in tissue_lower:
                return value
                
        raise ValueError(
            f"Unknown tissue: {tissue}. "
            f"Available: {', '.join(sorted(TISSUE_ONTOLOGY_MAP.keys()))}"
        )
    
    def _resolve_assays(self, assays: List[str]) -> List[Any]:
        """Convert assay names to OutputType enums."""
        output_types = []
        for assay in assays:
            assay_lower = assay.lower().strip()
            
            # Map alias to official name
            official_name = ASSAY_ALIASES.get(assay_lower, assay.upper())
            
            # Get the enum
            if hasattr(dna_client.OutputType, official_name):
                output_types.append(getattr(dna_client.OutputType, official_name))
            else:
                logger.warning(f"Unknown assay type: {assay}")
                
        return output_types
    
    def _parse_location(self, location: str) -> tuple:
        """Parse location string to (chrom, start, end)."""
        # Remove commas and spaces
        clean = location.replace(",", "").replace(" ", "").strip()
        
        if ":" not in clean:
            raise ValueError(f"Invalid location format: {location}. Use 'chr:start-end'")
        
        chrom, rest = clean.split(":", 1)
        
        if "-" in rest:
            start_str, end_str = rest.split("-", 1)
            start = int(start_str)
            end = int(end_str)
        else:
            # Single position
            pos = int(rest)
            start = pos
            end = pos + 1
            
        return chrom, start, end
    
    def _extract_region_stats(
        self, 
        values: np.ndarray,
        query_interval: Any,
        output_interval: Any,
        resolution: int = 1
    ) -> Dict[str, float]:
        """
        Extract statistics for query region from output values.
        
        Args:
            values: Track values array (seq_len, n_tracks) or (seq_len,)
            query_interval: The original query interval
            output_interval: The interval the output covers
            resolution: Output resolution in bp
            
        Returns:
            Dict with mean, max, min, std, n_bins
        """
        # Calculate query region indices within output
        query_start_in_output = query_interval.start - output_interval.start
        query_end_in_output = query_interval.end - output_interval.start
        
        # Convert to bin indices
        start_bin = max(0, query_start_in_output // resolution)
        end_bin = min(len(values), query_end_in_output // resolution)
        
        if start_bin >= end_bin:
            end_bin = min(len(values), start_bin + 1)
        
        # Handle multi-track arrays
        if values.ndim == 2:
            slice_data = values[start_bin:end_bin, :]
            # Average across tracks if multiple
            slice_data = np.nanmean(slice_data, axis=1)
        else:
            slice_data = values[start_bin:end_bin]
        
        # Remove NaNs
        slice_data = slice_data[~np.isnan(slice_data)]
        
        if len(slice_data) == 0:
            return {"mean": 0.0, "max": 0.0, "min": 0.0, "std": 0.0, "n_bins": 0}
        
        return {
            "mean": float(np.mean(slice_data)),
            "max": float(np.max(slice_data)),
            "min": float(np.min(slice_data)),
            "std": float(np.std(slice_data)),
            "n_bins": len(slice_data),
        }
    
    def predict_region(
        self,
        location: str,
        tissue: str = "lung",
        assays: Optional[List[str]] = None,
        sequence_length: str = "1MB",
        filter_strand: Optional[str] = None,
        filter_tf: Optional[str] = None,
        filter_histone: Optional[str] = None,
        compare_tissues: Optional[List[str]] = None,
        top_tracks: int = 10,
    ) -> Dict[str, Any]:
        """
        Predict functional signals for a genomic region.
        
        Args:
            location: Genomic coordinates "chr:start-end" (GRCh38)
            tissue: Tissue/cell type name or ontology ID
            assays: List of assay types (default: ["atac", "dnase", "rna"])
            sequence_length: Context window size ("2KB", "16KB", "100KB", "500KB", "1MB")
            filter_strand: Filter to specific strand ("+", "-", or None for both)
            filter_tf: Filter ChIP-TF to specific transcription factor
            filter_histone: Filter ChIP-Histone to specific mark (e.g., "H3K4me3")
            compare_tissues: Additional tissues to compare against
            top_tracks: For multi-track outputs, show top N by signal
            
        Returns:
            Dict with predictions, metadata, and interpretation guidance
        """
        if not self.client:
            return {"error": "AlphaGenome client not initialized. Check API key."}
        
        # Parse location
        try:
            chrom, start, end = self._parse_location(location)
        except ValueError as e:
            return {"error": str(e)}
        
        # Create query interval
        query_interval = genome.Interval(chromosome=chrom, start=start, end=end)
        
        # Get sequence length
        seq_len = self.SEQUENCE_LENGTHS.get(sequence_length.upper(), 1048576)
        
        # Resize to valid input length (centered on query)
        input_interval = query_interval.resize(seq_len)
        
        # Resolve tissue
        try:
            ontology_id = self._resolve_tissue(tissue)
        except ValueError as e:
            return {"error": str(e)}
        
        # Default assays
        if not assays:
            assays = ["atac", "dnase", "rna"]
        if isinstance(assays, str):
            assays = [a.strip() for a in assays.split(",")]
        
        # Resolve assay types
        output_types = self._resolve_assays(assays)
        if not output_types:
            return {"error": f"No valid assay types. Available: {list(ASSAY_ALIASES.keys())}"}
        
        # Build result
        result = {
            "query": {
                "location": f"{chrom}:{start:,}-{end:,}",
                "size_bp": end - start,
                "tissue": tissue,
                "ontology_id": ontology_id,
                "assays": [ot.name for ot in output_types],
                "context_window": sequence_length,
            },
            "predictions": {},
            "interpretation": {},
            "metadata": {},
        }
        
        try:
            # Make prediction
            output = self.client.predict_interval(
                interval=input_interval,
                requested_outputs=set(output_types),
                ontology_terms=[ontology_id],
            )
            
            # Process each output type
            for output_type in output_types:
                attr_name = output_type.name.lower()
                type_info = OUTPUT_TYPE_INFO.get(output_type.name, None)
                
                if not hasattr(output, attr_name):
                    result["predictions"][attr_name] = {"error": "Not available for this tissue"}
                    continue
                
                track_data = getattr(output, attr_name)
                values = track_data.values  # Shape: (seq_len, n_tracks)
                metadata = track_data.metadata
                
                # Apply filters
                track_mask = np.ones(len(metadata), dtype=bool)
                
                if filter_strand and "strand" in metadata.columns:
                    track_mask &= (metadata["strand"] == filter_strand).values
                
                if filter_tf and "transcription_factor" in metadata.columns:
                    track_mask &= metadata["transcription_factor"].str.contains(
                        filter_tf, case=False, na=False
                    ).values
                
                if filter_histone and "histone_mark" in metadata.columns:
                    track_mask &= metadata["histone_mark"].str.contains(
                        filter_histone, case=False, na=False
                    ).values
                
                # Filter values and metadata
                filtered_values = values[:, track_mask]
                filtered_metadata = metadata[track_mask]
                
                if filtered_values.shape[1] == 0:
                    result["predictions"][attr_name] = {"error": "No tracks match filters"}
                    continue
                
                # Get resolution
                resolution = type_info.resolution_bp if type_info else 1
                
                # Calculate statistics for query region
                stats = self._extract_region_stats(
                    filtered_values,
                    query_interval,
                    track_data.interval,
                    resolution
                )
                
                # Per-track statistics for top tracks
                track_stats = []
                max_signals = filtered_values.max(axis=0)
                top_indices = np.argsort(max_signals)[-top_tracks:][::-1]
                
                for idx in top_indices:
                    track_meta = filtered_metadata.iloc[idx].to_dict()
                    track_values = filtered_values[:, idx]
                    track_stat = self._extract_region_stats(
                        track_values,
                        query_interval,
                        track_data.interval,
                        resolution
                    )
                    track_stats.append({
                        "metadata": {k: v for k, v in track_meta.items() 
                                    if k in ["name", "biosample_name", "strand", 
                                            "transcription_factor", "histone_mark"]},
                        "stats": track_stat,
                    })
                
                result["predictions"][attr_name] = {
                    "aggregate": stats,
                    "n_tracks": filtered_values.shape[1],
                    "top_tracks": track_stats,
                    "units": type_info.units if type_info else "unknown",
                    "resolution_bp": resolution,
                }
                
                # Add interpretation
                if type_info:
                    result["interpretation"][attr_name] = type_info.interpretation
            
            # Compare tissues if requested
            if compare_tissues:
                result["tissue_comparison"] = {}
                for comp_tissue in compare_tissues:
                    if comp_tissue.lower() == tissue.lower():
                        continue
                    try:
                        comp_ontology = self._resolve_tissue(comp_tissue)
                        comp_output = self.client.predict_interval(
                            interval=input_interval,
                            requested_outputs=set(output_types),
                            ontology_terms=[comp_ontology],
                        )
                        
                        comp_results = {}
                        for output_type in output_types:
                            attr_name = output_type.name.lower()
                            if hasattr(comp_output, attr_name):
                                track_data = getattr(comp_output, attr_name)
                                type_info = OUTPUT_TYPE_INFO.get(output_type.name)
                                resolution = type_info.resolution_bp if type_info else 1
                                stats = self._extract_region_stats(
                                    track_data.values,
                                    query_interval,
                                    track_data.interval,
                                    resolution
                                )
                                comp_results[attr_name] = stats
                        
                        result["tissue_comparison"][comp_tissue] = comp_results
                        
                    except Exception as e:
                        result["tissue_comparison"][comp_tissue] = {"error": str(e)}
            
        except Exception as e:
            result["error"] = f"Prediction failed: {str(e)}"
            logger.exception("AlphaGenome prediction error")
        
        return result
    
    def predict_variant(
        self,
        location: str,
        ref_allele: str,
        alt_allele: str,
        tissue: str = "lung",
        assays: Optional[List[str]] = None,
        sequence_length: str = "1MB",
    ) -> Dict[str, Any]:
        """
        Predict the effect of a genetic variant.
        
        Uses the official AlphaGenome variant prediction API which:
        - Compares predictions between REF and ALT sequences
        - Handles indels correctly with proper alignment
        - Returns both raw scores and quantile scores
        
        Args:
            location: Variant position "chr:position" (1-based, GRCh38)
            ref_allele: Reference allele (e.g., "A", "ACGT")
            alt_allele: Alternate allele
            tissue: Tissue/cell type
            assays: Assay types to evaluate
            sequence_length: Context window size
            
        Returns:
            Dict with REF/ALT predictions, differences, and interpretation
        """
        if not self.client:
            return {"error": "AlphaGenome client not initialized"}
        
        # Parse position
        try:
            chrom, start, end = self._parse_location(location)
            position = start  # For single position
        except ValueError as e:
            return {"error": str(e)}
        
        # Validate alleles
        ref_allele = ref_allele.upper().strip()
        alt_allele = alt_allele.upper().strip()
        valid_bases = set("ACGTN")
        
        if not all(b in valid_bases for b in ref_allele):
            return {"error": f"Invalid reference allele: {ref_allele}"}
        if not all(b in valid_bases for b in alt_allele):
            return {"error": f"Invalid alternate allele: {alt_allele}"}
        
        # Create variant object (note: position is 1-based in Variant!)
        variant = genome.Variant(
            chromosome=chrom,
            position=position,
            reference_bases=ref_allele,
            alternate_bases=alt_allele,
        )
        
        # Create interval centered on variant
        seq_len = self.SEQUENCE_LENGTHS.get(sequence_length.upper(), 1048576)
        interval = variant.reference_interval.resize(seq_len)
        
        # Resolve tissue
        try:
            ontology_id = self._resolve_tissue(tissue)
        except ValueError as e:
            return {"error": str(e)}
        
        # Default assays for variant effect
        if not assays:
            assays = ["atac", "dnase", "rna", "cage"]
        if isinstance(assays, str):
            assays = [a.strip() for a in assays.split(",")]
        
        output_types = self._resolve_assays(assays)
        if not output_types:
            return {"error": "No valid assay types"}
        
        result = {
            "variant": {
                "location": f"{chrom}:{position}",
                "ref": ref_allele,
                "alt": alt_allele,
                "tissue": tissue,
            },
            "effects": {},
            "interpretation": [],
        }
        
        try:
            # Use official predict_variant API
            output = self.client.predict_variant(
                interval=interval,
                variant=variant,
                requested_outputs=set(output_types),
                ontology_terms=[ontology_id],
            )
            
            # Compare REF and ALT predictions at variant position
            # Region of interest: ±500bp around variant
            var_interval = genome.Interval(
                chromosome=chrom,
                start=max(0, position - 500),
                end=position + 500
            )
            
            for output_type in output_types:
                attr_name = output_type.name.lower()
                type_info = OUTPUT_TYPE_INFO.get(output_type.name)
                resolution = type_info.resolution_bp if type_info else 1
                
                ref_data = getattr(output.reference, attr_name, None)
                alt_data = getattr(output.alternate, attr_name, None)
                
                if ref_data is None or alt_data is None:
                    result["effects"][attr_name] = {"error": "Not available"}
                    continue
                
                # Extract stats for both
                ref_stats = self._extract_region_stats(
                    ref_data.values, var_interval, ref_data.interval, resolution
                )
                alt_stats = self._extract_region_stats(
                    alt_data.values, var_interval, alt_data.interval, resolution
                )
                
                # Calculate differences
                diff = alt_stats["mean"] - ref_stats["mean"]
                if ref_stats["mean"] > 1e-6:
                    fold_change = alt_stats["mean"] / ref_stats["mean"]
                    log2fc = np.log2(fold_change) if fold_change > 0 else 0
                else:
                    fold_change = float("inf") if alt_stats["mean"] > 1e-6 else 1.0
                    log2fc = 0
                
                # Interpret effect
                if abs(log2fc) < 0.1:
                    effect_label = "NEUTRAL"
                elif log2fc > 0:
                    effect_label = f"↑ INCREASED ({fold_change:.2f}x)"
                else:
                    effect_label = f"↓ DECREASED ({fold_change:.2f}x)"

                result["effects"][attr_name] = {
                    "ref_mean": ref_stats["mean"],
                    "alt_mean": alt_stats["mean"],
                    "difference": diff,
                    "fold_change": fold_change,
                    "log2_fold_change": log2fc,
                    "effect": effect_label,
                    "units": type_info.units if type_info else "unknown",
                }
            
            # Add interpretation guidance
            result["interpretation"] = [
                "⚠️ Predictions are relative, not absolute measurements",
                "• Large changes (|Δ| > 0.01) may indicate regulatory variants",
                "• ATAC/DNase changes suggest altered chromatin accessibility",
                "• CAGE changes suggest altered transcription initiation",
                "• ChIP changes suggest altered protein binding",
                "• Validate significant predictions experimentally",
                "• Use quantile scores for cross-assay comparisons",
            ]
            
        except Exception as e:
            result["error"] = f"Variant prediction failed: {str(e)}"
            logger.exception("AlphaGenome variant prediction error")
        
        return result
    
    def get_available_tracks(
        self,
        output_type: str,
        tissue: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get available tracks for an output type.
        
        Useful for discovering what TFs, histone marks, or tissues are available.
        
        Args:
            output_type: The output type (e.g., "chip_tf", "chip_histone")
            tissue: Optional tissue filter
            
        Returns:
            Dict with available tracks and their metadata
        """
        if not self.output_metadata:
            return {"error": "Output metadata not loaded"}
        
        # Get the metadata dataframe
        output_types = self._resolve_assays([output_type])
        if not output_types:
            return {"error": f"Unknown output type: {output_type}"}
        
        attr_name = output_types[0].name.lower()
        if not hasattr(self.output_metadata, attr_name):
            return {"error": f"No metadata for {output_type}"}
        
        metadata_df = getattr(self.output_metadata, attr_name)
        
        # Filter by tissue if specified
        if tissue:
            try:
                ontology_id = self._resolve_tissue(tissue)
                if "ontology_curie" in metadata_df.columns:
                    metadata_df = metadata_df[
                        metadata_df["ontology_curie"] == ontology_id
                    ]
            except ValueError:
                pass  # Keep all if tissue not found
        
        result = {
            "output_type": output_types[0].name,
            "total_tracks": len(metadata_df),
            "columns": list(metadata_df.columns),
        }
        
        # Summarize key columns
        if "biosample_name" in metadata_df.columns:
            result["biosamples"] = sorted(metadata_df["biosample_name"].unique().tolist())
        
        if "transcription_factor" in metadata_df.columns:
            tfs = metadata_df["transcription_factor"].dropna().unique()
            result["transcription_factors"] = sorted(tfs.tolist())
            result["n_tfs"] = len(tfs)
        
        if "histone_mark" in metadata_df.columns:
            marks = metadata_df["histone_mark"].dropna().unique()
            result["histone_marks"] = sorted(marks.tolist())
        
        if "strand" in metadata_df.columns:
            result["strands"] = metadata_df["strand"].unique().tolist()
        
        return result


# ============================================================================
# FORMAT OUTPUT FOR AGENT USE
# ============================================================================

def format_prediction_report(result: Dict[str, Any]) -> str:
    """Format prediction result as a readable report."""
    if "error" in result and not result.get("predictions"):
        return f"❌ Error: {result['error']}"
    
    lines = []
    lines.append("=" * 60)
    lines.append("**AlphaGenome Prediction Report**")
    lines.append("=" * 60)
    
    # Query info
    q = result.get("query", {})
    lines.append(f"Region: {q.get('location', 'N/A')} ({q.get('size_bp', 0):,} bp)")
    lines.append(f"Tissue: {q.get('tissue', 'N/A')} ({q.get('ontology_id', 'N/A')})")
    lines.append(f"Assays: {', '.join(q.get('assays', []))}")
    lines.append(f"Context: {q.get('context_window', 'N/A')}")
    lines.append("")
    
    # Predictions
    lines.append("**Predictions:**")
    lines.append("-" * 40)
    
    for assay, data in result.get("predictions", {}).items():
        if "error" in data:
            lines.append(f"• {assay.upper()}: {data['error']}")
            continue
        
        agg = data.get("aggregate", {})
        lines.append(f"• {assay.upper()}:")
        lines.append(f"    Mean: {agg.get('mean', 0):.4f}")
        lines.append(f"    Max:  {agg.get('max', 0):.4f}")
        lines.append(f"    Range: [{agg.get('min', 0):.4f} - {agg.get('max', 0):.4f}]")
        lines.append(f"    ({data.get('n_tracks', 0)} tracks, {agg.get('n_bins', 0)} bins × {data.get('resolution_bp', 1)}bp)")
        lines.append(f"    Units: {data.get('units', 'N/A')}")
        
        # Top tracks
        top = data.get("top_tracks", [])[:3]
        if top:
            lines.append("    Top tracks:")
            for t in top:
                meta = t.get("metadata", {})
                name = meta.get("biosample_name", meta.get("name", "unnamed"))
                extra = ""
                if "transcription_factor" in meta:
                    extra = f" [{meta['transcription_factor']}]"
                elif "histone_mark" in meta:
                    extra = f" [{meta['histone_mark']}]"
                lines.append(f"      - {name}{extra}: {t['stats']['mean']:.4f}")
    
    # Interpretation
    lines.append("")
    lines.append("**Interpretation Guide:**")
    for assay, interp in result.get("interpretation", {}).items():
        lines.append(f"• {assay.upper()}: {interp}")
    
    # Tissue comparison
    if result.get("tissue_comparison"):
        lines.append("")
        lines.append("**Tissue Comparison:**")
        for tissue, data in result["tissue_comparison"].items():
            if "error" in data:
                lines.append(f"  {tissue}: {data['error']}")
            else:
                tissue_line = f"  {tissue}: "
                parts = [f"{k}={v.get('mean', 0):.4f}" for k, v in data.items()]
                tissue_line += ", ".join(parts)
                lines.append(tissue_line)
    
    if "error" in result:
        lines.append("")
        lines.append(f"⚠️ Warning: {result['error']}")
    
    return "\n".join(lines)


def format_variant_report(result: Dict[str, Any]) -> str:
    """Format variant effect result as a readable report."""
    if "error" in result and not result.get("effects"):
        return f"❌ Error: {result['error']}"
    
    lines = []
    lines.append("=" * 60)
    lines.append("**AlphaGenome Variant Effect Prediction**")
    lines.append("=" * 60)
    
    v = result.get("variant", {})
    lines.append(f"Variant: {v.get('location', 'N/A')} {v.get('ref', '?')}>{v.get('alt', '?')}")
    lines.append(f"Tissue: {v.get('tissue', 'N/A')}")
    lines.append("")
    
    lines.append("**Predicted Effects (ALT vs REF):**")
    lines.append("-" * 40)
    
    for assay, data in result.get("effects", {}).items():
        if "error" in data:
            lines.append(f"• {assay.upper()}: {data['error']}")
            continue
        
        lines.append(f"• {assay.upper()}:")
        lines.append(f"    REF: {data.get('ref_mean', 0):.4f}")
        lines.append(f"    ALT: {data.get('alt_mean', 0):.4f}")
        lines.append(f"    Δ = {data.get('difference', 0):+.4f} ({data.get('fold_change', 1):.2f}x) → {data.get('effect', 'NEUTRAL')}")
    
    lines.append("")
    lines.append("**Interpretation:**")
    for interp in result.get("interpretation", []):
        lines.append(interp)
    
    return "\n".join(lines)


# ============================================================================
# LANGCHAIN TOOLS (if using with an agent)
# ============================================================================

try:
    from langchain_core.tools import tool
    
    # Global handler instance
    _ag_handler = None
    
    def _get_handler() -> AlphaGenomeHandler:
        global _ag_handler
        if _ag_handler is None:
            _ag_handler = AlphaGenomeHandler()
        return _ag_handler
    
    @tool
    def alphagenome_predict(
        location: str,
        tissue: str = "lung",
        assays: str = "atac,dnase,rna",
        compare_tissues: str = "",
        filter_tf: str = "",
        filter_histone: str = "",
    ) -> str:
        """
        Predict functional genomic signals for a region using AlphaGenome AI.
        
        Best for: Predicting chromatin accessibility, transcription, TF binding,
        histone modifications, and splicing at any genomic region.
        
        Args:
            location: Genomic coordinates "chr:start-end" (GRCh38).
                     Example: "chr17:7676000-7677000" (TP53 promoter)
            tissue: Target tissue/cell type. Options include:
                   Tissues: lung, liver, brain, heart, kidney, colon, breast, etc.
                   Cell lines: k562, hepg2 (BEST for TF binding - 269/501 TFs!)
                   Cell types: hepatocyte, neuron, t_cell, etc.
            assays: Comma-separated assay types:
                   - atac, dnase: Chromatin accessibility
                   - rna, cage, procap: Gene expression/TSS
                   - histone: Histone modifications (H3K4me3, H3K27ac, etc.)
                   - tf: Transcription factor binding (CTCF, etc.)
                   - splice, splice_usage, junctions: Splicing
                   - contacts: 3D chromatin (Hi-C)
            compare_tissues: Optional comma-separated tissues to compare
            filter_tf: Filter ChIP-TF to specific TF (e.g., "CTCF")
            filter_histone: Filter ChIP-Histone to specific mark (e.g., "H3K4me3")
            
        Returns:
            Prediction report with signal values and interpretation guidance.
            
        Note:
            - Predictions are RELATIVE, not absolute measurements
            - Best for COMPARING regions or tissues
            - ChIP outputs are fold-change over control (>1 = enriched)
            - Cell lines K562/HepG2 have best TF coverage
        """
        handler = _get_handler()
        if not handler.client:
            return "❌ AlphaGenome not available. Check API key (ALPHAGENOME_API_KEY)."
        
        assay_list = [a.strip() for a in assays.split(",") if a.strip()]
        compare_list = [t.strip() for t in compare_tissues.split(",") if t.strip()] or None
        
        result = handler.predict_region(
            location=location,
            tissue=tissue,
            assays=assay_list,
            compare_tissues=compare_list,
            filter_tf=filter_tf or None,
            filter_histone=filter_histone or None,
        )
        
        return format_prediction_report(result)
    
    @tool
    def alphagenome_variant_effect(
        location: str,
        ref_allele: str,
        alt_allele: str,
        tissue: str = "lung",
        assays: str = "atac,dnase,rna,cage",
    ) -> str:
        """
        Predict the regulatory effect of a genetic variant using AlphaGenome.
        
        Compares predictions between reference and alternate alleles to
        identify potential regulatory variants affecting chromatin, expression, etc.
        
        Args:
            location: Variant position "chr:position" (1-based, GRCh38)
                     Example: "chr17:7676100"
            ref_allele: Reference allele (e.g., "A", "ACGT" for indels)
            alt_allele: Alternate allele (e.g., "G", "T")
            tissue: Target tissue for prediction
            assays: Assay types to evaluate effects on
            
        Returns:
            Report showing REF vs ALT predictions and predicted effect direction.
            
        Example:
            alphagenome_variant_effect(
                location="chr17:7676100",
                ref_allele="C",
                alt_allele="T", 
                tissue="lung"
            )
        """
        handler = _get_handler()
        if not handler.client:
            return "❌ AlphaGenome not available. Check API key."
        
        assay_list = [a.strip() for a in assays.split(",") if a.strip()]
        
        result = handler.predict_variant(
            location=location,
            ref_allele=ref_allele,
            alt_allele=alt_allele,
            tissue=tissue,
            assays=assay_list,
        )
        
        return format_variant_report(result)
    
    @tool
    def alphagenome_available_tracks(
        output_type: str,
        tissue: str = "",
    ) -> str:
        """
        List available tracks for an AlphaGenome output type.
        
        Use this to discover what transcription factors, histone marks,
        or tissues are available for prediction.
        
        Args:
            output_type: The output type to query:
                        - "tf" or "chip_tf": List available TFs
                        - "histone" or "chip_histone": List histone marks
                        - "atac", "dnase", "rna", etc.: List tissues
            tissue: Optional tissue filter (especially useful for TF)
            
        Returns:
            List of available tracks with metadata.
            
        Example:
            # See what TFs are available in K562
            alphagenome_available_tracks(output_type="tf", tissue="k562")
            
            # See what histone marks are available
            alphagenome_available_tracks(output_type="histone")
        """
        handler = _get_handler()
        if not handler.client:
            return "❌ AlphaGenome not available."
        
        result = handler.get_available_tracks(
            output_type=output_type,
            tissue=tissue or None,
        )
        
        if "error" in result:
            return f"❌ {result['error']}"
        
        lines = [f"**Available tracks for {result['output_type']}:**"]
        lines.append(f"Total tracks: {result['total_tracks']}")
        
        if "transcription_factors" in result:
            lines.append(f"\nTranscription Factors ({result['n_tfs']}):")
            lines.append(", ".join(result["transcription_factors"][:20]))
            if result["n_tfs"] > 20:
                lines.append(f"  ... and {result['n_tfs'] - 20} more")
        
        if "histone_marks" in result:
            lines.append(f"\nHistone Marks: {', '.join(result['histone_marks'])}")
        
        if "biosamples" in result:
            lines.append(f"\nBiosamples ({len(result['biosamples'])}):")
            lines.append(", ".join(result["biosamples"][:10]))
            if len(result["biosamples"]) > 10:
                lines.append(f"  ... and {len(result['biosamples']) - 10} more")
        
        return "\n".join(lines)

except ImportError:
    # LangChain not available - that's fine, handler still works
    pass


# ============================================================================
# MAIN / TESTING
# ============================================================================

if __name__ == "__main__":
    # Quick test
    handler = AlphaGenomeHandler()
    
    if handler.client:
        print("Testing AlphaGenome prediction...")
        result = handler.predict_region(
            location="chr17:7676000-7677000",  # TP53 region
            tissue="lung",
            assays=["atac", "dnase", "rna"],
        )
        print(format_prediction_report(result))
    else:
        print("AlphaGenome client not initialized. Set ALPHAGENOME_API_KEY.")
