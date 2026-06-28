"""
Shared biomedical tool library.
===============================
Gene annotation, gene–disease associations, genomic coordinates, AlphaGenome
predictions, and expression (bulk + single-cell). Pathway/interaction tools live
in pathway_tools.py. All HTTP goes through the shared cached/retrying session.

Imported by unified_agent.py (and usable standalone).
"""

import os
os.environ.setdefault("ATLASAPPROX_HIDECREDITS", "yes")

import io
import logging
from functools import lru_cache
from typing import List, Optional

import requests
import urllib3
import pandas as pd
from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# m6A-Atlas (rnamd.org) serves over HTTPS with an invalid certificate, so its
# tool must call with verify=False; silence the resulting per-request warning.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import http_client
from opentargets_tool import query_gene_diseases
from gene_resolver import resolve_gene, resolve_symbol

# Optional heavy deps — wrap so import never hard-fails.
try:
    import atlasapprox
    ATLASAPPROX_AVAILABLE = True
except ImportError:
    ATLASAPPROX_AVAILABLE = False

try:
    from alphagenome.models import dna_client  # noqa: F401
    from alphagenome.data import genome         # noqa: F401
    ALPHAGENOME_AVAILABLE = True
except ImportError:
    ALPHAGENOME_AVAILABLE = False

from alphagenome_tool import AlphaGenomePredictor

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 15

load_dotenv()
ALPHAGENOME_API_KEY = os.getenv("ALPHAGENOME_API_KEY")


# =========================================
# GENE INFO  (NCBI Gene + UniProt)
# =========================================

@tool
def gene_info_tool(gene_symbol: str) -> str:
    """Get gene function, aliases, diseases from MyGene.info+UniProt. Use FIRST for any gene question."""
    query = gene_symbol.strip()

    output = [f"**Gene Information: {query.upper()}**\n"]
    sources_status = {"MyGene.info": "❌ NOT QUERIED", "UniProt": "❌ NOT QUERIED"}
    found_info = False
    official_symbol = query.upper()

    # 1. MyGene.info (resolves aliases, consolidates NCBI/Ensembl/UniProt)
    g = resolve_gene(query)
    if g:
        found_info = True
        sources_status["MyGene.info"] = "✅ SUCCESS"
        official_symbol = g.get("symbol", official_symbol)
        output.append(f"**Official Symbol:** {official_symbol}")
        if g.get("name"):
            output.append(f"**Full Name:** {g['name']}")
        alias = g.get("alias")
        if alias:
            alias = [alias] if isinstance(alias, str) else alias
            output.append(f"**Aliases:** {', '.join(alias)}")
        summary = g.get("summary", "")
        if summary:
            output.append(f"\n**Function:**\n{summary[:600] + ('...' if len(summary) > 600 else '')}")
        gp = g.get("genomic_pos")
        if isinstance(gp, list):
            gp = gp[0] if gp else None
        if isinstance(gp, dict) and gp.get("chr"):
            output.append(f"\n**Location:** Chromosome {gp.get('chr')}")
        if g.get("entrezgene"):
            output.append(f"**NCBI Gene ID:** {g['entrezgene']}")
        ens = g.get("ensembl")
        if isinstance(ens, list):
            ens = ens[0] if ens else None
        if isinstance(ens, dict) and ens.get("gene"):
            output.append(f"**Ensembl ID:** {ens['gene']}")
    else:
        sources_status["MyGene.info"] = "⚠️ GENE NOT FOUND"

    # 2. UniProt (detailed protein function + disease) — query by resolved symbol
    try:
        uniprot_url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            "query": f"gene_exact:{official_symbol} AND organism_id:9606 AND reviewed:true",
            "fields": "accession,protein_name,cc_function,cc_disease",
            "format": "json",
            "size": 1
        }
        r = http_client.get(uniprot_url, params=params, headers={"Accept": "application/json"}, timeout=REQUEST_TIMEOUT)

        if r.ok:
            results = r.json().get("results", [])
            if results:
                found_info = True
                sources_status["UniProt"] = "✅ SUCCESS"
                entry = results[0]
                protein_name = entry.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "")
                if protein_name:
                    output.append(f"\n**Protein Name:** {protein_name}")
                accession = entry.get("primaryAccession", "")
                if accession:
                    output.append(f"**UniProt ID:** {accession}")
                for comment in entry.get("comments", []):
                    if comment.get("commentType") == "FUNCTION":
                        texts = comment.get("texts", [])
                        if texts:
                            func_text = texts[0].get("value", "")
                            if func_text and "Function:" not in "\n".join(output):
                                output.append(f"\n**Protein Function:**\n{func_text[:400] + ('...' if len(func_text) > 400 else '')}")
                    if comment.get("commentType") == "DISEASE":
                        disease_name = comment.get("disease", {}).get("diseaseId", "")
                        if disease_name:
                            output.append(f"\n**Associated Disease:** {disease_name}")
                            break
            else:
                sources_status["UniProt"] = "⚠️ GENE NOT FOUND"
        else:
            sources_status["UniProt"] = f"❌ FAILED (HTTP {r.status_code})"
    except Exception as e:
        sources_status["UniProt"] = f"❌ ERROR: {str(e)[:50]}"

    output.append("\n" + "=" * 50)
    output.append("**DATA SOURCE STATUS:**")
    for source, status in sources_status.items():
        output.append(f"  • {source}: {status}")

    if not found_info:
        output.append("\n❌ **NO AUTHORITATIVE DATA RETRIEVED**")
        output.append(f"Gene '{gene_symbol}' was not found in any database.")
        output.append("⚠️ DO NOT make claims about this gene - admit you cannot find information.")

    return "\n".join(output)


# =========================================
# GENE→DISEASE ASSOCIATIONS  (Open Targets)
# =========================================

@tool
def opentargets_associations_tool(gene_symbol: str) -> str:
    """Get diseases associated with a gene (with evidence scores) from Open Targets."""
    return query_gene_diseases(gene_symbol)


# =========================================
# GENE COORDINATES  (Ensembl)
# =========================================

@tool
def get_gene_coordinates_tool(gene_symbol: str) -> str:
    """Get genomic coordinates (GRCh38) for a gene from Ensembl."""
    gene_symbol = gene_symbol.strip().upper()
    output = []
    sources_status = {"Ensembl REST API": "❌ NOT QUERIED"}

    try:
        url = f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene_symbol}"
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        r = http_client.get(url, headers=headers, timeout=REQUEST_TIMEOUT)

        if r.ok:
            data = r.json()
            chrom = data.get("seq_region_name", "Unknown")
            strand = "+" if data.get("strand", 1) == 1 else "-"
            sources_status["Ensembl REST API"] = "✅ SUCCESS"
            output.append(f"**{gene_symbol}** coordinates (GRCh38):")
            output.append(f"  Chromosome: {chrom}")
            output.append(f"  Start: {data.get('start', 0)}")
            output.append(f"  End: {data.get('end', 0)}")
            output.append(f"  Strand: {strand}")
            output.append(f"  Ensembl ID: {data.get('id', 'N/A')}")
        elif r.status_code == 404:
            sources_status["Ensembl REST API"] = "⚠️ GENE NOT FOUND"
            output.append(f"Gene '{gene_symbol}' not found in Ensembl")
        else:
            sources_status["Ensembl REST API"] = f"❌ HTTP {r.status_code}"
            output.append(f"Ensembl API error: HTTP {r.status_code}")
    except Exception as e:
        sources_status["Ensembl REST API"] = "❌ ERROR"
        output.append(f"Error: {str(e)}")

    output.append("\n" + "=" * 50)
    output.append("**DATA SOURCE STATUS:**")
    for source, status in sources_status.items():
        output.append(f"  • {source}: {status}")
    return "\n".join(output)


def get_promoter_region(gene_symbol: str, upstream: int = 1500, downstream: int = 500) -> dict:
    """Get promoter coordinates for a gene."""
    gene_symbol = gene_symbol.strip().upper()
    try:
        url = f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene_symbol}"
        r = http_client.get(url, headers={"Content-Type": "application/json"}, timeout=15)
        if r.ok:
            data = r.json()
            chrom = data.get("seq_region_name")
            start = data.get("start")
            end = data.get("end")
            strand = data.get("strand", 1)
            if strand == 1:
                tss = start
                promoter_start, promoter_end = tss - upstream, tss + downstream
            else:
                tss = end
                promoter_start, promoter_end = tss - downstream, tss + upstream
            return {
                "gene": gene_symbol, "tss": tss,
                "strand": "+" if strand == 1 else "-",
                "promoter_location": f"chr{chrom}:{promoter_start}-{promoter_end}",
                "promoter_size": upstream + downstream,
            }
        return {"error": f"Gene not found: {gene_symbol}"}
    except Exception as e:
        return {"error": str(e)}


@tool
def get_promoter_coordinates_tool(gene_symbol: str, upstream: int = 1500, downstream: int = 500) -> str:
    """Get promoter region coordinates for AlphaGenome predictions."""
    result = get_promoter_region(gene_symbol, upstream, downstream)
    if "error" in result:
        return f"Error: {result['error']}"
    return (
        f"**Promoter Region for {result['gene']}**\n"
        f"  TSS position: {result['tss']:,}\n"
        f"  Strand: {result['strand']}\n"
        f"  Promoter: {result['promoter_location']}\n"
        f"  Size: {result['promoter_size']} bp\n\n"
        f"Use this location for AlphaGenome: {result['promoter_location']}"
    )


# =========================================
# ALPHAGENOME PREDICTION
# =========================================

class GenomicPredictionInput(BaseModel):
    coordinates: str = Field(description="Genomic coordinates in 'chr:start-end' format (e.g., 'chr17:7676000-7677000').")
    tissue: str = Field(description="Target tissue or cell type name (e.g., 'lung', 'liver', 'T-cell').")
    assays: List[str] = Field(default=["atac"], description="Assays: ['atac', 'dnase', 'rna', 'cage', 'chip_histone', 'chip_tf'].")


_PREDICTOR: Optional[AlphaGenomePredictor] = None


def _get_predictor() -> AlphaGenomePredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = AlphaGenomePredictor(skip_validation=False)
    return _PREDICTOR


@tool(args_schema=GenomicPredictionInput)
def predict_genomics(coordinates: str, tissue: str, assays: List[str] = ["atac"]) -> str:
    """Predict genomic signals (ATAC/RNA/ChIP) for coordinates+tissue via AlphaGenome."""
    if not ALPHAGENOME_AVAILABLE:
        return "AlphaGenome library is not installed; genomic prediction is unavailable."
    if not ALPHAGENOME_API_KEY:
        return "ALPHAGENOME_API_KEY is not set; cannot run genomic predictions."
    try:
        result = _get_predictor().predict_coordinates(
            coordinates=coordinates, tissue=tissue, assays=assays, sequence_length="auto"
        )
        from alphagenome_tool import format_results
        return format_results(result)
    except Exception as e:
        return f"Error running AlphaGenome prediction: {str(e)}"


# =========================================
# BULK TISSUE EXPRESSION  (GTEx)
# =========================================

@tool
def gene_tissue_expression_tool(gene_symbol: str, tissue: str = None) -> str:
    """Query bulk tissue expression (TPM) from GTEx v10. Returns median TPM across tissues."""
    input_symbol = gene_symbol.strip()
    # Normalize aliases (e.g. "p53" -> "TP53") so GTEx resolves the gene.
    resolved_symbol = resolve_symbol(input_symbol) or input_symbol.upper()
    output = ["**GTEx Tissue Expression Analysis (Bulk RNA-seq)**", f"Gene: {resolved_symbol}"]
    if resolved_symbol.upper() != input_symbol.upper():
        output.append(f"(resolved from '{input_symbol}')")
    if tissue:
        output.append(f"Tissue focus: {tissue}")
    output.append("")

    gtex_id = None
    official_symbol = resolved_symbol.upper()
    try:
        params = {
            "geneId": resolved_symbol, "gencodeVersion": "v39",
            "genomeBuild": "GRCh38/hg38", "page": 0, "itemsPerPage": 10
        }
        r_gtex = http_client.get("https://gtexportal.org/api/v2/reference/gene",
                                 params=params, headers={"Accept": "application/json"}, timeout=10)
        if r_gtex.ok:
            gene_data = r_gtex.json().get("data", [])
            if gene_data:
                best = gene_data[0]
                gtex_id = best.get("gencodeId")
                official_symbol = best.get("geneSymbol", input_symbol.upper())
                output.append(f"✅ Resolved: {official_symbol}")
                output.append(f"   GTEx ID: {gtex_id}")
                if best.get("description"):
                    output.append(f"   Description: {best.get('description')}")
                output.append("")
            else:
                output.append(f"⚠️  '{input_symbol}' not found in GTEx reference. Try the official HGNC symbol or Ensembl ID.")
                output.append("")
        else:
            output.append(f"⚠️  GTEx gene lookup failed (HTTP {r_gtex.status_code}).")
            output.append("")
    except requests.exceptions.Timeout:
        return "\n".join(output + ["❌ Error: GTEx gene lookup timed out."])
    except Exception as e:
        return "\n".join(output + [f"❌ Error resolving gene: {str(e)[:100]}"])

    if not gtex_id:
        output.append("❌ Cannot query expression without valid GTEx ID.")
        output.append("- Try the Ensembl ID, or use gene_info_tool to find the official symbol")
        return "\n".join(output)

    try:
        exp_params = {"gencodeId": gtex_id, "datasetId": "gtex_v10", "format": "json"}
        output.append("Querying GTEx v10 expression data...")
        r_exp = http_client.get("https://gtexportal.org/api/v2/expression/medianGeneExpression",
                                params=exp_params, headers={"Accept": "application/json"}, timeout=15)
        if not r_exp.ok:
            return "\n".join(output + [f"❌ GTEx API error: HTTP {r_exp.status_code}"])

        expressions = r_exp.json().get("data", [])
        if not expressions:
            return "\n".join(output + [f"⚠️  No expression data returned for {official_symbol}."])

        output.append(f"✅ Found expression data across {len(expressions)} tissues\n")
        sorted_exp = sorted(expressions, key=lambda x: x.get("median", 0), reverse=True)

        if tissue:
            tissue_lower = tissue.lower()
            filtered = [e for e in sorted_exp if tissue_lower in e.get("tissueSiteDetailId", "").lower()]
            if filtered:
                output.append(f"**Expression in '{tissue}' tissues:**")
                output.append(f"{'Tissue':<35} {'Median TPM':>12} {'n':>5}")
                output.append(f"{'-'*35} {'-'*12} {'-'*5}")
                for e in filtered:
                    output.append(f"{e.get('tissueSiteDetailId', 'Unknown')[:34]:<35} "
                                  f"{e.get('median', 0):>11.2f} {str(e.get('nSamples', 'N/A')):>5}")
            else:
                output.append(f"⚠️  No tissues matching '{tissue}'. Examples:")
                for e in sorted_exp[:5]:
                    output.append(f"   • {e.get('tissueSiteDetailId')}")
        else:
            output.append("**Tissues by Expression:**")
            output.append(f"{'Rank':<5} {'Tissue':<35} {'Median TPM':>12} {'n':>5}")
            output.append(f"{'-'*5} {'-'*35} {'-'*12} {'-'*5}")
            for rank, e in enumerate(sorted_exp, 1):
                output.append(f"{rank:<5} {e.get('tissueSiteDetailId', 'Unknown')[:34]:<35} "
                              f"{e.get('median', 0):>11.2f} {str(e.get('nSamples', 'N/A')):>5}")
            output.append(f"\n  Highest: {sorted_exp[0].get('tissueSiteDetailId')} ({sorted_exp[0].get('median'):.2f} TPM)")
            output.append(f"  Lowest:  {sorted_exp[-1].get('tissueSiteDetailId')} ({sorted_exp[-1].get('median'):.2f} TPM)")

        output.append("\n• TPM = Transcripts Per Million; bulk RNA-seq = average across all cells in tissue")
        all_medians = [e.get("median", 0) for e in expressions]
        top_tpm = sorted_exp[0].get("median", 0)
        median_tpm = sorted(all_medians)[len(all_medians) // 2] if all_medians else 0
        if top_tpm > 10 * median_tpm and top_tpm > 10:
            output.append(f"• **Tissue-specific**: high in {sorted_exp[0].get('tissueSiteDetailId')}")
        elif median_tpm > 1:
            output.append("• **Broadly expressed** across most tissues")
        else:
            output.append("• **Low expression** across tissues")
        return "\n".join(output)
    except requests.exceptions.Timeout:
        return "\n".join(output + ["❌ Error: GTEx expression query timed out."])
    except Exception as e:
        return "\n".join(output + [f"❌ Error querying GTEx: {str(e)[:100]}"])


# =========================================
# SINGLE-CELL EXPRESSION  (AtlasApprox)
# =========================================

# AtlasApprox organ/organism lists are static — cache them so we don't make an
# extra round-trip on every query (the AtlasApprox client isn't behind http_client).
_ORGANISM_ALIASES = {
    "human": "h_sapiens", "homo sapiens": "h_sapiens", "h sapiens": "h_sapiens",
    "mouse": "m_musculus", "mus musculus": "m_musculus",
}


@lru_cache(maxsize=1)
def _atlas_api():
    return atlasapprox.API()


@lru_cache(maxsize=1)
def _atlas_organisms() -> tuple:
    """Organisms available for gene-expression measurements."""
    data = _atlas_api().organisms()
    if isinstance(data, dict):
        ge = data.get("gene_expression")
        if ge:
            return tuple(ge)
        return tuple(sorted({o for v in data.values() for o in v}))
    return tuple(data)


@lru_cache(maxsize=16)
def _atlas_organs(organism: str) -> tuple:
    return tuple(_atlas_api().organs(organism=organism))


def _atlas_celltypes(organism: str, organ: str) -> tuple:
    try:
        return tuple(_atlas_api().celltypes(organism=organism, organ=organ))
    except Exception:
        return ()


@tool
def get_cell_type_markers(organ: str, cell_type: str, organism: str = "h_sapiens", number: int = 15) -> str:
    """Marker genes that define a CELL TYPE within an organ (AtlasApprox). Works for any organism; defaults to human."""
    if not ATLASAPPROX_AVAILABLE:
        return "AtlasApprox is not installed; cell-type markers are unavailable."
    organism = _ORGANISM_ALIASES.get(organism.lower().strip(), organism)
    try:
        if organism not in _atlas_organisms():
            return (f"❌ '{organism}' is not a valid AtlasApprox organism. "
                    f"Use e.g. 'h_sapiens' or 'm_musculus'.")
        match = next((o for o in _atlas_organs(organism) if o.lower() == organ.lower()), None)
        if not match:
            return (f"❌ '{organ}' is not a valid organ for {organism}. Choose from:\n"
                    f"  {', '.join(_atlas_organs(organism))}")
        markers = _atlas_api().markers(organism=organism, organ=match, cell_type=cell_type, number=number)
        if not markers:
            cts = _atlas_celltypes(organism, match)
            return (f"No markers returned for cell type '{cell_type}' in {match} ({organism}). "
                    f"Valid cell types: {', '.join(cts)}" if cts else
                    f"No markers returned for '{cell_type}' in {match} ({organism}).")
        return (f"**Marker genes for '{cell_type}' in {match} ({organism}) — AtlasApprox**\n"
                f"{', '.join(markers)}")
    except Exception as e:
        cts = _atlas_celltypes(organism, organ)
        hint = f" Valid cell types in {organ}: {', '.join(cts)}" if cts else ""
        return f"❌ Error querying AtlasApprox markers: {str(e)[:120]}.{hint}"


@tool
def gene_highest_expression_celltype(gene_symbol: str, organism: str = "h_sapiens", number: int = 10) -> str:
    """RANK cell types by HIGHEST expression of a gene across all organs (reverse lookup, AtlasApprox). Use ONLY for superlative 'most/highest/top-expressing' questions — NOT for general 'which cell types express X'. Defaults to human."""
    if not ATLASAPPROX_AVAILABLE:
        return "AtlasApprox is not installed; this lookup is unavailable."
    organism = _ORGANISM_ALIASES.get(organism.lower().strip(), organism)
    try:
        if organism not in _atlas_organisms():
            return (f"❌ '{organism}' is not a valid AtlasApprox organism. "
                    f"Use e.g. 'h_sapiens' or 'm_musculus'.")
        resolved = resolve_symbol(gene_symbol) or gene_symbol
        series = _atlas_api().highest_measurement(organism=organism, feature=resolved, number=number)
        lines = [f"**Cell types with highest expression of {resolved} ({organism}) — AtlasApprox**",
                 f"{'Cell type (organ)':<40} {'avg expression':>14}",
                 f"{'-'*40} {'-'*14}"]
        for idx, val in series.items():
            celltype, organ = idx if isinstance(idx, tuple) else (idx, "")
            lines.append(f"{f'{celltype} ({organ})'[:39]:<40} {float(val):>14.1f}")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Error querying AtlasApprox: {str(e)[:150]} (check the gene symbol)."


@tool
def get_gene_fraction_detected(gene_symbol: str, organ: str, organism: str = "h_sapiens") -> str:
    """NON-HUMAN single-cell expression (fraction of cells) within an ORGAN from AtlasApprox (e.g. mouse, zebrafish). For HUMAN single-cell expression use hpa_protein_atlas_tool. The 'organ' arg is a whole organ/tissue, NOT a cell type."""
    if not ATLASAPPROX_AVAILABLE:
        return "AtlasApprox is not installed; single-cell expression is unavailable."
    organism = _ORGANISM_ALIASES.get(organism.lower().strip(), organism)
    try:
        # Validate organism first (cached), for a clean error instead of a raw exception.
        valid_organisms = _atlas_organisms()
        if organism not in valid_organisms:
            return (
                f"❌ '{organism}' is not a valid AtlasApprox organism.\n"
                f"Use e.g. 'h_sapiens' (human) or 'm_musculus' (mouse). Available:\n"
                f"  {', '.join(valid_organisms)}"
            )

        # Validate organ against the available list (cached — no redundant round-trip).
        valid_organs = _atlas_organs(organism)
        match = next((o for o in valid_organs if o.lower() == organ.lower()), None)
        if not match:
            # Single, explicit, non-retry message: the model passed something that
            # isn't an organ (often a cell type). Tell it to pick from the list.
            return (
                f"❌ '{organ}' is not a valid organ for organism '{organism}'.\n"
                f"This tool takes a whole ORGAN, not a cell type — it returns the "
                f"per-cell-type breakdown within that organ.\n"
                f"Choose ONE organ from this list (do not retry with a cell type):\n"
                f"  {', '.join(valid_organs)}\n"
                f"For immune/blood cell types (e.g. T cells, macrophages), use 'blood' or 'immune'."
            )

        # Normalize gene aliases so the atlas lookup succeeds.
        resolved = resolve_symbol(gene_symbol) or gene_symbol
        df = pd.DataFrame(_atlas_api().fraction_detected(organism=organism, organ=match, features=[resolved]))
        lookup = resolved if resolved in df.index else (gene_symbol if gene_symbol in df.index else None)
        if lookup is None:
            return (f"⚠️ No single-cell data for '{gene_symbol}' in {match} ({organism}). "
                    f"The symbol may be unrecognized in this atlas.")

        top_20 = df.loc[lookup].sort_values(ascending=False).head(20)
        lines = [
            "**AtlasApprox Single-Cell Expression (fraction of cells detecting gene)**",
            f"Gene: {resolved}", f"Organ: {match} ({organism})", "",
            f"{'Cell type':<35} {'% cells detected':>16}",
            f"{'-'*35} {'-'*16}",
        ]
        for cell_type, fraction in top_20.items():
            lines.append(f"{str(cell_type)[:34]:<35} {fraction * 100:>15.1f}%")
        lines.append("")
        lines.append("Note: fraction_detected = proportion of cells of that type expressing the gene.")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Error querying AtlasApprox: {str(e)[:150]}"


# =========================================
# GEO DATASET AVAILABILITY  (NCBI E-utilities, db=gds)
# =========================================

# Common DataSet-type filters, mapped from loose user phrasing to GEO's exact
# "DataSet Type" vocabulary so the [DataSet Type] field tag actually matches.
_GEO_TYPE_ALIASES = {
    "rna-seq": "Expression profiling by high throughput sequencing",
    "rnaseq": "Expression profiling by high throughput sequencing",
    "scrna-seq": "Expression profiling by high throughput sequencing",
    "single-cell": "Expression profiling by high throughput sequencing",
    "microarray": "Expression profiling by array",
    "array": "Expression profiling by array",
    "atac-seq": "Genome binding/occupancy profiling by high throughput sequencing",
    "chip-seq": "Genome binding/occupancy profiling by high throughput sequencing",
    "methylation": "Methylation profiling by high throughput sequencing",
}


@tool
def geo_search_tool(query: str, organism: str = "", study_type: str = "",
                    max_results: int = 10) -> str:
    """Check NCBI GEO (Gene Expression Omnibus) for the AVAILABILITY of public studies/datasets
    matching a topic, disease, gene, or tissue. Returns GSE accessions, titles, organism,
    sample counts, dates, and links — NOT expression values. Use for "is there a dataset/study
    on X", "find GEO data for X", "are there RNA-seq studies of disease Y". For actual expression
    levels use the GTEx / HPA / AtlasApprox tools instead.

    organism: optional, e.g. "Homo sapiens" or "Mus musculus" (restricts to that species).
    study_type: optional, e.g. "rna-seq", "microarray", "atac-seq", "chip-seq", "methylation".
    """
    term = query.strip()
    if not term:
        return "Provide a search topic (disease, gene, tissue, or keywords)."
    if organism:
        term += f' AND "{organism.strip()}"[Organism]'
    if study_type:
        gds_type = _GEO_TYPE_ALIASES.get(study_type.lower().strip(), study_type.strip())
        term += f' AND "{gds_type}"[DataSet Type]'

    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    try:
        es = http_client.get_ncbi(
            base + "esearch.fcgi",
            params={"db": "gds", "term": term, "retmax": max(1, min(max_results, 50)),
                    "retmode": "json"},
        ).json()
        ids = es.get("esearchresult", {}).get("idlist", [])
        total = es.get("esearchresult", {}).get("count", "0")
        if not ids:
            return (f"No GEO studies found for: {query}"
                    + (f" (organism={organism})" if organism else "")
                    + (f" (type={study_type})" if study_type else "")
                    + ".")

        summ = http_client.get_ncbi(
            base + "esummary.fcgi",
            params={"db": "gds", "id": ",".join(ids), "retmode": "json"},
        ).json().get("result", {})

        lines = [f"**GEO study availability for: {query}** "
                 f"(showing {len(ids)} of {total} matches)\n"]
        for uid in ids:
            rec = summ.get(uid)
            if not isinstance(rec, dict):
                continue
            acc = rec.get("accession", "").strip()
            title = rec.get("title", "(no title)").strip()
            taxon = rec.get("taxon", "").strip()
            n = rec.get("n_samples", "")
            gtype = rec.get("gdstype", "").strip()
            date = rec.get("pdat", "").strip()
            link = (f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc}"
                    if acc.startswith("GSE") else "")
            lines.append(f"- **{acc or uid}** — {title}")
            meta = []
            if taxon:
                meta.append(taxon)
            if n:
                meta.append(f"{n} samples")
            if gtype:
                meta.append(gtype)
            if date:
                meta.append(date)
            if meta:
                lines.append(f"  {' | '.join(meta)}")
            if link:
                lines.append(f"  {link}")
        lines.append("\nNote: lists study availability/metadata only — not expression values.")
        lines.append("If a link returns a 500/502 error, retry — these are transient NCBI "
                     "server errors; the accession can also be pasted into "
                     "https://www.ncbi.nlm.nih.gov/geo/ to find the study manually.")
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Error querying GEO: {str(e)[:150]}"


# =========================================
# m6A RNA METHYLATION  (m6A-Atlas v2.0, rnamd.org)
# =========================================

# Map common organism names to m6A-Atlas's CamelCase scientific-name vocabulary.
_M6A_SPECIES = {
    "human": "HomoSapiens", "homo sapiens": "HomoSapiens", "h_sapiens": "HomoSapiens",
    "mouse": "MusMusculus", "mus musculus": "MusMusculus", "m_musculus": "MusMusculus",
    "rat": "RattusNorvegicus", "rattus norvegicus": "RattusNorvegicus",
    "zebrafish": "DanioRerio", "danio rerio": "DanioRerio",
    "yeast": "SaccharomycesCerevisiae", "saccharomyces cerevisiae": "SaccharomycesCerevisiae",
    "arabidopsis": "ArabidopsisThaliana", "arabidopsis thaliana": "ArabidopsisThaliana",
}


@tool
def m6a_modification_tool(gene_symbol: str, organism: str = "human", resolution: str = "high",
                          tissue: str = "") -> str:
    """Check the m6A-Atlas for N6-methyladenosine (m6A) RNA methylation on a GENE: whether the
    gene's transcript carries m6A modifications and WHERE (transcript region + genomic
    coordinates), with supporting evidence. Use for "does GENE have m6A / is GENE m6A-methylated /
    where is GENE methylated / m6A sites on GENE". This is RNA methylation (epitranscriptomics),
    not DNA methylation. resolution='high' returns single-base sites (default); 'low' returns
    MeRIP-seq peak regions with differential signal (Log2FC). Human by default.

    tissue: optional cell-line or tissue name to restrict the sites to (e.g. "HepG2", "HeLa",
    "liver", "heart"). Matched case-insensitively against m6A-Atlas's own cell-line/tissue
    vocabulary; if nothing matches, the reply lists which cell lines/tissues ARE available for
    that gene. Use it for "does GENE have m6A in TISSUE / m6A sites of GENE in TISSUE".
    """
    species = _M6A_SPECIES.get(organism.lower().strip())
    if not species:
        return (f"❌ '{organism}' is not a supported m6A-Atlas species. "
                f"Supported: {', '.join(sorted(set(_M6A_SPECIES.values())))}.")
    symbol = (resolve_symbol(gene_symbol) or gene_symbol.strip()).upper()
    high = resolution.lower().strip() != "low"
    endpoint = "apiHighResolution.php" if high else "apiLowResolution.php"
    res_label = "single-base resolution" if high else "MeRIP-seq peak resolution"

    try:
        resp = http_client.get(
            f"https://www.rnamd.org/m6a/{endpoint}",
            params={"species": species, "gene": symbol},
            verify=False, timeout=30,  # invalid TLS cert; data is public read-only
        )
        text = resp.text or ""
        if not text.strip():
            return f"No response from m6A-Atlas for {symbol} ({species})."
        df = pd.read_csv(io.StringIO(text), sep="\t")
    except Exception as e:
        return f"❌ Error querying m6A-Atlas: {str(e)[:150]}"

    # The 'gene' filter is server-side, but keep only exact GeneName matches as a
    # safety net against any loose/partial matching.
    if "GeneName" in df.columns:
        df = df[df["GeneName"].astype(str).str.upper() == symbol]
    if df.empty:
        return (f"No m6A sites reported for {symbol} in {species} "
                f"(m6A-Atlas v2.0, {res_label}).")

    # Optional tissue/cell-line filter. Done client-side (case-insensitive substring)
    # so it works for high-resolution too, which has no server-side cell-line param.
    # The CellLine field can be a ';'-joined list of cell lines per site.
    if tissue and "CellLine" in df.columns:
        want = tissue.strip()
        matched = df[df["CellLine"].astype(str).str.contains(want, case=False, na=False, regex=False)]
        if matched.empty:
            avail = sorted({c.strip() for cell in df["CellLine"].astype(str)
                            for c in cell.split(";") if c.strip()})
            return (f"No m6A sites for {symbol} in tissue/cell-line matching '{tissue}' "
                    f"({species}, {res_label}). Available for this gene: "
                    f"{', '.join(avail[:30])}{' …' if len(avail) > 30 else ''}.")
        df = matched

    n = len(df)
    scope = f" in {tissue}" if tissue else ""
    title = "m6A sites" if high else "m6A MeRIP-seq peaks"
    lines = [f"**{title} for {symbol} ({species}){scope} — m6A-Atlas v2.0, {res_label}**",
             f"Total: {n}\n"]

    # Transcript-region breakdown (the 'where', categorically).
    if "Region" in df.columns:
        lines.append("By transcript region:")
        for region, count in df["Region"].value_counts().items():
            lines.append(f"  {region}: {count}")
        lines.append("")

    if high:
        # Rank single-base sites by how many conditions/datasets support them.
        support_col = "ConditionNum" if "ConditionNum" in df.columns else None
        view = df.sort_values(support_col, ascending=False) if support_col else df
        lines.append(f"Top sites (by supporting conditions):")
        lines.append(f"{'Position':<24} {'Strand':<6} {'Region':<14} {'#cond':>6}")
        lines.append(f"{'-'*24} {'-'*6} {'-'*14} {'-'*6}")
        for _, r in view.head(10).iterrows():
            pos = f"{r.get('Seqname', '?')}:{r.get('Position', '?')}"
            support = r.get(support_col, "") if support_col else ""
            lines.append(f"{pos[:24]:<24} {str(r.get('Strand', '')):<6} "
                         f"{str(r.get('Region', ''))[:14]:<14} {str(support):>6}")
    else:
        # Rank MeRIP peaks by enrichment (Log2FC) when available.
        if "Log2FC" in df.columns:
            df["Log2FC"] = pd.to_numeric(df["Log2FC"], errors="coerce")
            view = df.sort_values("Log2FC", ascending=False)
        else:
            view = df
        lines.append("Top peaks (by Log2FC enrichment):")
        lines.append(f"{'Region (chr:start-end)':<30} {'Region':<14} {'Log2FC':>8} {'CellLine':<14}")
        lines.append(f"{'-'*30} {'-'*14} {'-'*8} {'-'*14}")
        for _, r in view.head(10).iterrows():
            loc = f"{r.get('Seqname', '?')}:{r.get('Start', '?')}-{r.get('End', '?')}"
            fc = r.get("Log2FC", "")
            fc = f"{fc:.2f}" if isinstance(fc, float) and pd.notna(fc) else str(fc)
            lines.append(f"{loc[:30]:<30} {str(r.get('Region', ''))[:14]:<14} "
                         f"{fc:>8} {str(r.get('CellLine', ''))[:14]:<14}")

    lines.append("\nSource: m6A-Atlas v2.0 (rnamd.org). m6A is RNA (not DNA) methylation.")
    return "\n".join(lines)


# All non-pathway tools, for convenient import.
BIO_TOOLS = [
    gene_info_tool,
    opentargets_associations_tool,
    get_gene_coordinates_tool,
    get_promoter_coordinates_tool,
    predict_genomics,
    gene_tissue_expression_tool,
    get_gene_fraction_detected,
    get_cell_type_markers,
    gene_highest_expression_celltype,
    geo_search_tool,
    m6a_modification_tool,
]
