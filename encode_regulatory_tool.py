"""
ENCODE Regulatory Elements Tool v2
===================================
Query ENCODE/SCREEN for tissue-specific regulatory element activity.

API Options:
1. SCREEN REST API (screen-beta-api.wenglab.org) - for coordinate searches
2. SCREEN GraphQL API (api.wenglab.org/screen_graphql/graphql) - for cCRE details
3. ENCODE Portal REST API - fallback for experiment metadata

cCRE Types:
- PLS: Promoter-like signature (H3K4me3 high, near TSS)
- pELS: Proximal enhancer-like signature (H3K27ac high, near gene)
- dELS: Distal enhancer-like signature (H3K27ac high, far from gene)
- CA-CTCF: Chromatin accessible with CTCF binding (insulator)
- CA-H3K4me3: Chromatin accessible with H3K4me3 (not near annotated TSS)
"""

import requests
import json
import uuid
from typing import Optional, List, Dict, Any

try:
    from langchain_core.tools import tool
except ImportError:
    def tool(func):
        return func

REQUEST_TIMEOUT = 30

# ============================================
# API ENDPOINTS
# ============================================

# SCREEN REST API for coordinate-based searches
SCREEN_REST_API = "https://screen-beta-api.wenglab.org/dataws/cre_table"

# SCREEN GraphQL API (alternative URL from GitHub issues)  
SCREEN_GRAPHQL_API = "https://api.wenglab.org/screen_graphql/graphql"

# ENCODE Portal
ENCODE_PORTAL_URL = "https://www.encodeproject.org"


# ============================================
# SCREEN REST API FUNCTIONS
# ============================================

def search_ccres_by_region(
    chromosome: str,
    start: int,
    end: int,
    assembly: str = "GRCh38"
) -> Optional[Dict]:
    """
    Search for cCREs in a genomic region using SCREEN REST API.
    """
    # Normalize chromosome
    if not chromosome.startswith("chr"):
        chromosome = f"chr{chromosome}"
    
    payload = {
        "uuid": str(uuid.uuid4()),
        "assembly": assembly,
        "accessions": [],
        "coord_chrom": chromosome,
        "coord_start": start,
        "coord_end": end,
        # Include all element types
        "gene_all_start": 0,
        "gene_all_end": 5000000,
        "gene_pc_start": 0,
        "gene_pc_end": 5000000,
        # Signal thresholds (1.64 = 95th percentile)
        "rank_dnase_start": -10,
        "rank_dnase_end": 10,
        "rank_promoter_start": -10,
        "rank_promoter_end": 10,
        "rank_enhancer_start": -10,
        "rank_enhancer_end": 10,
        "rank_ctcf_start": -10,
        "rank_ctcf_end": 10,
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    try:
        r = requests.post(
            SCREEN_REST_API,
            json=payload,
            headers=headers,
            timeout=REQUEST_TIMEOUT
        )
        if r.ok:
            return r.json()
        else:
            return {"error": f"HTTP {r.status_code}: {r.text[:200]}"}
    except requests.exceptions.Timeout:
        return {"error": "SCREEN API timeout"}
    except Exception as e:
        return {"error": str(e)}


def get_ccre_details_graphql(accession: str, assembly: str = "GRCh38") -> Optional[Dict]:
    """
    Get detailed cCRE information using GraphQL API.
    """
    # Query for single cCRE by accession
    query = """
    query cCREDetails($accession: String!, $assembly: String!) {
        ccre(accession: $accession, assembly: $assembly) {
            accession
            coordinates {
                chromosome
                start
                end
            }
            group
            dnase
            h3k4me3
            h3k27ac
            ctcf
        }
    }
    """
    
    variables = {
        "accession": accession,
        "assembly": assembly
    }
    
    try:
        r = requests.post(
            SCREEN_GRAPHQL_API,
            json={"query": query, "variables": variables},
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT
        )
        if r.ok:
            return r.json()
        return {"error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"error": str(e)}


# ============================================
# ENSEMBL API
# ============================================

def get_gene_coordinates(gene_symbol: str) -> Optional[Dict[str, Any]]:
    """Get gene coordinates from Ensembl."""
    try:
        url = f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene_symbol.upper()}"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        
        if r.ok:
            data = r.json()
            return {
                "chromosome": data.get("seq_region_name"),
                "start": data.get("start"),
                "end": data.get("end"),
                "strand": "+" if data.get("strand", 1) == 1 else "-",
                "ensembl_id": data.get("id")
            }
        return None
    except:
        return None


# ============================================
# HELPER FUNCTIONS
# ============================================

def get_ccre_group_description(group: str) -> str:
    """Get human-readable description of cCRE group."""
    descriptions = {
        "PLS": "Promoter-Like Signature",
        "pELS": "Proximal Enhancer-Like Signature", 
        "dELS": "Distal Enhancer-Like Signature",
        "CA-CTCF": "CTCF-bound (Insulator)",
        "CA-H3K4me3": "Open Chromatin with H3K4me3",
        "CA-TF": "TF-bound Chromatin Accessible",
        "CA": "Chromatin Accessible Only",
        "TF": "TF-bound"
    }
    return descriptions.get(group, group)


def interpret_zscore(zscore: float) -> str:
    """Interpret Z-score activity level."""
    if zscore is None:
        return "N/A"
    if zscore >= 1.64:
        return "HIGH"
    elif zscore >= 0:
        return "mod"
    else:
        return "low"


# ============================================
# LANGCHAIN TOOLS
# ============================================

@tool
def regulatory_elements_tool(
    gene_symbol: Optional[str] = None,
    chromosome: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    flank_kb: int = 50,
    assembly: str = "GRCh38"
) -> str:
    """
    Search for regulatory elements (enhancers, promoters, insulators) near a gene or region.
    
    USE THIS TOOL FOR: "Does [gene] have H3K4me3/H3K27ac marks?" or "Find enhancers near [gene]"
    
    This tool uses ENCODE/SCREEN database of 2.3M+ candidate cis-Regulatory Elements (cCREs).
    It returns regulatory elements with their MAX Z-scores across all biosamples in ENCODE.
    
    IMPORTANT: The Z-scores are AGGREGATE across many tissues/cell types. They show whether
    a region has regulatory potential ANYWHERE, not tissue-specific activity.
    
    Args:
        gene_symbol: Gene name (e.g., "TP53", "DDR2", "EGFR"). Will search within flanking region.
        chromosome: Chromosome (e.g., "chr1", "1"). Use with start/end for custom region.
        start: Start coordinate (GRCh38)
        end: End coordinate (GRCh38)
        flank_kb: Kilobases to search upstream/downstream of gene (default: 50kb)
        assembly: Genome assembly (default: GRCh38)
    
    Returns:
        List of regulatory elements with their type, location, and activity levels (Z-scores).
        
    cCRE Types (what the Z-scores tell you):
        - PLS: Promoter-like (HIGH H3K4me3 near TSS) → This gene's promoter is active
        - pELS: Proximal enhancer (HIGH H3K27ac, <2kb from gene) → Active enhancer nearby
        - dELS: Distal enhancer (HIGH H3K27ac, >2kb from gene) → Distal regulatory element
        - CA-CTCF: Insulator element (CTCF bound) → Boundary element
        
    Interpreting Z-scores:
        - Z > 1.64: HIGH activity (95th percentile across all biosamples)
        - Z > 0: Moderate activity
        - Z < 0: Low activity
    """
    output = []
    sources_status = {
        "SCREEN REST API": "❌ NOT QUERIED",
        "Ensembl REST API": "❌ NOT QUERIED"
    }
    
    # Determine genomic region
    if gene_symbol:
        sources_status["Ensembl REST API"] = "⏳ QUERYING..."
        coords = get_gene_coordinates(gene_symbol)
        
        if coords:
            sources_status["Ensembl REST API"] = "✅ SUCCESS"
            chromosome = coords["chromosome"]
            # Add flanking region
            flank = flank_kb * 1000
            start = max(0, coords["start"] - flank)
            end = coords["end"] + flank
            
            output.append(f"**Searching regulatory elements near {gene_symbol.upper()}**")
            output.append(f"Region: chr{chromosome}:{start:,}-{end:,} (gene ± {flank_kb}kb)")
            output.append(f"Gene location: chr{chromosome}:{coords['start']:,}-{coords['end']:,}")
        else:
            sources_status["Ensembl REST API"] = "❌ GENE NOT FOUND"
            output.append(f"❌ Could not find coordinates for gene '{gene_symbol}'")
            output.append("\n" + "="*50)
            output.append("**DATA SOURCE STATUS:**")
            for source, status in sources_status.items():
                output.append(f"  • {source}: {status}")
            return "\n".join(output)
    
    elif chromosome and start and end:
        output.append(f"**Searching regulatory elements in region**")
        output.append(f"Region: chr{chromosome}:{start:,}-{end:,}")
    else:
        return "❌ Please provide either a gene_symbol OR chromosome/start/end coordinates"
    
    # Query SCREEN REST API
    sources_status["SCREEN REST API"] = "⏳ QUERYING..."
    
    result = search_ccres_by_region(chromosome, start, end, assembly)
    
    if result and "error" in result:
        sources_status["SCREEN REST API"] = f"❌ {result['error'][:50]}"
        output.append(f"\n❌ SCREEN API Error: {result['error']}")
    elif result:
        # Parse REST API response
        ccres = result.get("data", result.get("cres", result.get("results", [])))
        
        if not ccres and isinstance(result, list):
            ccres = result
        
        if ccres:
            sources_status["SCREEN REST API"] = "✅ SUCCESS"
            output.append(f"\n**Found {len(ccres)} candidate cis-Regulatory Elements (cCREs)**\n")
            
            # Group by type
            by_type = {}
            for ccre in ccres:
                # Handle different response formats
                group = ccre.get("group", ccre.get("ccre_group", ccre.get("type", "Unknown")))
                if group not in by_type:
                    by_type[group] = []
                by_type[group].append(ccre)
            
            # Summary by type
            output.append("**Summary by Type:**")
            for group, elements in sorted(by_type.items()):
                desc = get_ccre_group_description(group)
                output.append(f"  • {group} ({desc}): {len(elements)}")
            
            output.append("")
            
            # Show top elements
            output.append("**Regulatory Elements (sorted by activity):**")
            header = f"{'Accession':<18} {'Type':<10} {'Location':<28} {'DNase':>6} {'H3K27ac':>7} {'H3K4me3':>7} {'CTCF':>6}"
            output.append(header)
            output.append("-" * len(header))
            
            # Sort by max signal
            def get_max_signal(c):
                signals = []
                for key in ['dnase', 'h3k27ac', 'h3k4me3', 'ctcf', 'rank_dnase', 'rank_enhancer', 'rank_promoter', 'rank_ctcf']:
                    val = c.get(key)
                    if val is not None and isinstance(val, (int, float)):
                        signals.append(val)
                return max(signals) if signals else -10
            
            sorted_ccres = sorted(ccres, key=get_max_signal, reverse=True)
            
            for ccre in sorted_ccres[:25]:  # Show top 25
                acc = ccre.get("accession", ccre.get("ccre", "N/A"))[:17]
                group = ccre.get("group", ccre.get("ccre_group", "?"))[:9]
                
                # Get coordinates - handle different formats
                chrom = ccre.get("chrom", ccre.get("chromosome", "?"))
                if isinstance(chrom, str):
                    chrom = chrom.replace("chr", "")
                cstart = ccre.get("start", ccre.get("coord_start", 0))
                cend = ccre.get("stop", ccre.get("end", ccre.get("coord_end", 0)))
                loc = f"chr{chrom}:{cstart:,}-{cend:,}"[:27]
                
                # Get signals - handle different key names
                dnase = ccre.get("dnase", ccre.get("rank_dnase"))
                h3k27ac = ccre.get("h3k27ac", ccre.get("rank_enhancer"))
                h3k4me3 = ccre.get("h3k4me3", ccre.get("rank_promoter"))
                ctcf = ccre.get("ctcf", ccre.get("rank_ctcf"))
                
                def fmt_signal(s):
                    if s is None: return "N/A"
                    return f"{s:.1f}" if isinstance(s, float) else str(s)
                
                output.append(
                    f"{acc:<18} {group:<10} {loc:<28} "
                    f"{fmt_signal(dnase):>6} {fmt_signal(h3k27ac):>7} "
                    f"{fmt_signal(h3k4me3):>7} {fmt_signal(ctcf):>6}"
                )
            
            if len(ccres) > 25:
                output.append(f"\n... and {len(ccres) - 25} more elements")
            
            # Interpretation
            output.append("\n**Interpretation:**")
            output.append("• Z-scores > 1.64 indicate HIGH activity (95th percentile)")
            output.append("• PLS = Promoter-like (H3K4me3 near TSS)")
            output.append("• pELS/dELS = Enhancers (H3K27ac, proximal/distal)")
            output.append("• CA-CTCF = Potential insulator/boundary elements")
            
        else:
            sources_status["SCREEN REST API"] = "⚠️ NO DATA"
            output.append(f"\n⚠️ No regulatory elements found in this region")
            output.append(f"Raw response type: {type(result)}")
            if isinstance(result, dict):
                output.append(f"Keys: {list(result.keys())[:5]}")
    else:
        sources_status["SCREEN REST API"] = "⚠️ EMPTY RESPONSE"
        output.append("\n⚠️ Empty response from SCREEN API")
    
    # Links
    if gene_symbol:
        output.append(f"\n**Explore in SCREEN:**")
        output.append(f"https://screen.wenglab.org/search?q={gene_symbol}&assembly={assembly}")
    
    # Data source status
    output.append("\n" + "="*50)
    output.append("**DATA SOURCE STATUS:**")
    for source, status in sources_status.items():
        output.append(f"  • {source}: {status}")
    
    return "\n".join(output)


@tool
def encode_experiments_tool(
    biosample: str,
    assay_type: str = "all",
    target: Optional[str] = None
) -> str:
    """
    Search ENCODE portal for regulatory experiments in a specific tissue/cell type.
    
    USE THIS TOOL FOR: "What experiments exist for [tissue]?" or "Find ChIP-seq data in [cell line]"
    
    NOTE: For gene-specific questions like "Does DDR2 have H3K4me3 marks in brain?",
    use regulatory_elements_tool instead - it queries SCREEN cCREs near the gene.
    
    Args:
        biosample: Tissue or cell type. Common values:
            - Cell lines (best coverage): "K562", "HepG2", "GM12878", "A549", "MCF-7"
            - Tissues: "brain", "liver", "lung", "heart", "kidney"
            - Note: Human primary tissue experiments are LIMITED compared to cell lines
        assay_type: Type of assay:
            - "all": All regulatory assays
            - "DNase-seq": Chromatin accessibility
            - "ATAC-seq": Chromatin accessibility  
            - "H3K27ac": Active enhancer mark
            - "H3K4me3": Promoter mark
            - "CTCF": Insulator binding
            - "TF ChIP-seq": Transcription factor binding
        target: Specific histone mark or TF (e.g., "H3K27ac", "CTCF", "TP53")
    
    Returns:
        List of available ENCODE experiments for the specified tissue
    """
    output = []
    sources_status = {"ENCODE Portal API": "❌ NOT QUERIED"}
    
    output.append(f"**Searching ENCODE for regulatory experiments**")
    output.append(f"Biosample: {biosample}")
    if assay_type != "all":
        output.append(f"Assay type: {assay_type}")
    if target:
        output.append(f"Target: {target}")
    output.append("")
    
    # Map user-friendly assay names to ENCODE terms
    assay_mapping = {
        "all": None,
        "DNase-seq": "DNase-seq",
        "ATAC-seq": "ATAC-seq", 
        "H3K27ac": "Histone ChIP-seq",
        "H3K4me3": "Histone ChIP-seq",
        "CTCF": "TF ChIP-seq",
        "TF ChIP-seq": "TF ChIP-seq",
        "Histone ChIP-seq": "Histone ChIP-seq"
    }
    
    encode_assay = assay_mapping.get(assay_type, assay_type)
    
    try:
        # ENCODE API - build URL with correct parameters
        # Key insight: some parameters use different field paths
        base_url = f"{ENCODE_PORTAL_URL}/search/"
        
        params = {
            "type": "Experiment",
            "status": "released",
            "limit": "50",
            "format": "json"
        }
        
        # Biosample - try multiple field names for flexibility
        # "organ_slims" works better for general tissue names like "brain", "liver"
        # "biosample_ontology.term_name" is for exact cell type names
        if biosample.lower() in ["brain", "liver", "lung", "heart", "kidney", "spleen", "skin", "blood", "intestine", "stomach", "muscle"]:
            params["organ_slims"] = biosample
        else:
            params["biosample_ontology.term_name"] = biosample
        
        # Assay type
        if encode_assay:
            params["assay_title"] = encode_assay
            
        # Target (for ChIP-seq)
        if target:
            params["target.label"] = target
        elif assay_type in ["H3K27ac", "H3K4me3", "H3K4me1", "H3K27me3", "H3K36me3", "H3K9me3"]:
            params["target.label"] = assay_type
            
        headers = {
            "Accept": "application/json"
        }
        
        # Build query URL
        constructed_url = f"{base_url}?" + "&".join(f"{k}={v}" for k,v in params.items())
        
        r = requests.get(base_url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        
        # Update with actual URL used
        if hasattr(r, 'url'):
            constructed_url = r.url
        
        # If no results or error, try with searchTerm for flexible matching
        if r.status_code == 404 or (r.ok and len(r.json().get("@graph", [])) == 0):
            # Fallback: use searchTerm for more flexible matching
            params_fallback = {
                "type": "Experiment",
                "status": "released",
                "searchTerm": f"{biosample} {target or assay_type}".strip(),
                "limit": "50",
                "format": "json"
            }
            if encode_assay:
                params_fallback["assay_title"] = encode_assay
                
            r = requests.get(base_url, params=params_fallback, headers=headers, timeout=REQUEST_TIMEOUT)
            constructed_url = r.url if hasattr(r, 'url') else "fallback URL"
            if r.ok and len(r.json().get("@graph", [])) > 0:
                output.append(f"(Used flexible search for '{biosample}')")
        
        if r.ok:
            sources_status["ENCODE Portal API"] = "✅ SUCCESS"
            data = r.json()
            experiments = data.get("@graph", [])
            
            if not experiments:
                output.append(f"⚠️ No experiments found for '{biosample}' with current filters")
                output.append("\n**Why this might happen:**")
                if biosample.lower() in ["brain", "lung", "heart", "kidney"]:
                    output.append(f"• Human primary '{biosample}' tissue has LIMITED experiments in ENCODE")
                    output.append(f"• Most ENCODE data is from cell lines (K562, HepG2, GM12878)")
                    output.append(f"• For gene-specific questions, use regulatory_elements_tool instead")
                output.append("\n**Suggestions:**")
                output.append("• Try cell lines: 'K562', 'HepG2', 'GM12878', 'A549', 'MCF-7'")
                output.append("• Browse matrix: https://www.encodeproject.org/matrix/?type=Experiment")
                output.append(f"\n**Query used:** {constructed_url[:100]}...")
            else:
                output.append(f"**Found {len(experiments)} experiments**\n")
                
                # Group by assay type
                by_assay = {}
                for exp in experiments:
                    assay = exp.get("assay_title", "Unknown")
                    if assay not in by_assay:
                        by_assay[assay] = []
                    by_assay[assay].append(exp)
                
                for assay, exps in sorted(by_assay.items()):
                    output.append(f"\n**{assay}** ({len(exps)} experiments)")
                    
                    if "ChIP" in assay:
                        # Group by target
                        by_target = {}
                        for e in exps:
                            t = e.get("target", {})
                            if isinstance(t, dict):
                                t = t.get("label", "Unknown")
                            if t not in by_target:
                                by_target[t] = []
                            by_target[t].append(e)
                        
                        for t, texps in sorted(by_target.items())[:10]:
                            acc = texps[0].get("accession", "?")
                            output.append(f"  • {t}: {len(texps)} exp (e.g., {acc})")
                    else:
                        for e in exps[:5]:
                            acc = e.get("accession", "?")
                            bs = str(e.get("biosample_summary", "?"))[:50]
                            output.append(f"  • {acc}: {bs}")
                        if len(exps) > 5:
                            output.append(f"  ... and {len(exps) - 5} more")
                
                output.append(f"\n**Explore in ENCODE Portal:**")
                output.append(f"https://www.encodeproject.org/search/?type=Experiment&biosample_ontology.term_name={biosample.replace(' ', '+')}")
                
        else:
            sources_status["ENCODE Portal API"] = f"❌ HTTP {r.status_code}"
            output.append(f"❌ ENCODE API error: HTTP {r.status_code}")
            output.append(f"   Query URL: {constructed_url[:150]}...")
            # Try to get error details
            try:
                error_detail = r.json() if r.text else {}
                if "description" in error_detail:
                    output.append(f"   Details: {error_detail['description'][:100]}")
            except:
                if r.text:
                    output.append(f"   Response: {r.text[:100]}")
            output.append(f"\n**Suggestions:**")
            output.append(f"• For tissues use: 'brain', 'liver', 'lung', 'heart', 'kidney'")
            output.append(f"• For cell lines use: 'K562', 'HepG2', 'GM12878'")
            output.append(f"• Check: https://www.encodeproject.org/matrix/?type=Experiment")
            
    except requests.exceptions.Timeout:
        sources_status["ENCODE Portal API"] = "❌ TIMEOUT"
        output.append("❌ ENCODE API timeout")
    except Exception as e:
        sources_status["ENCODE Portal API"] = f"❌ ERROR"
        output.append(f"❌ Error: {str(e)[:100]}")
    
    # Data source status
    output.append("\n" + "="*50)
    output.append("**DATA SOURCE STATUS:**")
    for source, status in sources_status.items():
        output.append(f"  • {source}: {status}")
    
    return "\n".join(output)


# ============================================
# STANDALONE TESTING
# ============================================

if __name__ == "__main__":
    print("Testing ENCODE Regulatory Elements Tool v2\n")
    print("="*60)
    
    # Test 1: Query by gene
    print("\nTest 1: Search near TP53 (using REST API)")
    print("-" * 40)
    result = regulatory_elements_tool.invoke({
        "gene_symbol": "TP53",
        "flank_kb": 20
    })
    print(result)
    
    print("\n" + "="*60)
    
    # Test 2: ENCODE experiments
    print("\nTest 2: Search ENCODE for liver experiments")
    print("-" * 40)
    result = encode_experiments_tool.invoke({
        "biosample": "liver",
        "assay_type": "H3K27ac"
    })
    print(result)
