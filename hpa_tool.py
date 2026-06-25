"""
Human Protein Atlas (HPA) tool.
===============================
Human-only, precomputed, light to query (one cached fetch per gene). Surfaces the
HPA fields that aren't covered elsewhere in the agent:

  * Subcellular localization (immunofluorescence)
  * Cancer prognostic markers (TCGA survival correlation, 21 cohorts)
  * Brain regional expression
  * Immune / blood-cell expression + secretome (from protein class)
  * Tissue specificity classification

Resolves the gene symbol/alias to an Ensembl ID via the shared MyGene resolver,
then pulls the per-gene record from the HPA search-download API.

API: https://www.proteinatlas.org/  (no key required)
"""

import logging

from langchain_core.tools import tool

import http_client
from gene_resolver import resolve_gene

logger = logging.getLogger(__name__)

HPA_URL = "https://www.proteinatlas.org/api/search_download.php"

# 21 TCGA prognostic columns (codes from the HPA data-access docs).
_PROGNOSTIC_COLS = [
    "prognostic_Bladder_Urothelial_Carcinoma_(TCGA)",
    "prognostic_Breast_Invasive_Carcinoma_(TCGA)",
    "prognostic_Cervical_Squamous_Cell_Carcinoma_and_Endocervical_Adenocarcinoma_(TCGA)",
    "prognostic_Colon_Adenocarcinoma_(TCGA)",
    "prognostic_Glioblastoma_Multiforme_(TCGA)",
    "prognostic_Head_and_Neck_Squamous_Cell_Carcinoma_(TCGA)",
    "prognostic_Kidney_Chromophobe_(TCGA)",
    "prognostic_Kidney_Renal_Clear_Cell_Carcinoma_(TCGA)",
    "prognostic_Kidney_Renal_Papillary_Cell_Carcinoma_(TCGA)",
    "prognostic_Liver_Hepatocellular_Carcinoma_(TCGA)",
    "prognostic_Lung_Adenocarcinoma_(TCGA)",
    "prognostic_Lung_Squamous_Cell_Carcinoma_(TCGA)",
    "prognostic_Ovary_Serous_Cystadenocarcinoma_(TCGA)",
    "prognostic_Pancreatic_Adenocarcinoma_(TCGA)",
    "prognostic_Prostate_Adenocarcinoma_(TCGA)",
    "prognostic_Rectum_Adenocarcinoma_(TCGA)",
    "prognostic_Skin_Cutaneous_Melanoma_(TCGA)",
    "prognostic_Stomach_Adenocarcinoma_(TCGA)",
    "prognostic_Testicular_Germ_Cell_Tumor_(TCGA)",
    "prognostic_Thyroid_Carcinoma_(TCGA)",
    "prognostic_Uterine_Corpus_Endometrial_Carcinoma_(TCGA)",
]

_COLUMNS = [
    "g", "eg", "up", "pc",
    "rnats", "rnatd", "rnatsm",        # tissue specificity / distribution / specific nTPM
    "rnascs", "rnascd", "rnascsm",     # single cell type specificity / distribution / specific nCPM
    "rnabrs", "rnabrsm",               # brain regional
    "rnabcs", "rnabcsm",               # blood cell
    "scml", "scal",                    # subcellular main / additional
] + _PROGNOSTIC_COLS


def _ensembl_id(gene_symbol: str):
    g = resolve_gene(gene_symbol)
    if not g:
        return None, None
    ens = g.get("ensembl")
    if isinstance(ens, list):
        ens = ens[0] if ens else None
    ensg = ens.get("gene") if isinstance(ens, dict) else None
    return ensg, g.get("symbol")


def _fetch(ensg: str):
    try:
        r = http_client.get(
            HPA_URL,
            params={"search": ensg, "format": "json",
                    "columns": ",".join(_COLUMNS), "compress": "no"},
        )
        if r.ok:
            data = r.json()
            for rec in data:                 # prefer exact Ensembl match
                if rec.get("Ensembl") == ensg:
                    return rec
            return data[0] if data else None
        logger.warning("HPA HTTP %s", r.status_code)
    except Exception as e:
        logger.warning("HPA request failed: %s", e)
    return None


def _fmt_ntpm(v):
    """RNA specific nTPM fields are dicts like {'placenta': '61.8'} (or None)."""
    if isinstance(v, dict) and v:
        return ", ".join(f"{k} ({val})" for k, val in v.items())
    return None


@tool
def hpa_protein_atlas_tool(gene_symbol: str) -> str:
    """Human Protein Atlas for a HUMAN gene: subcellular localization, cancer prognostic markers, brain-regional & immune/blood expression, tissue specificity, secretome."""
    ensg, sym = _ensembl_id(gene_symbol)
    if not ensg:
        return f"Could not resolve '{gene_symbol}' to an Ensembl gene ID for HPA (human only)."

    rec = _fetch(ensg)
    if not rec:
        return f"No Human Protein Atlas data found for {sym or gene_symbol} ({ensg})."

    out = [f"**Human Protein Atlas — {rec.get('Gene', sym)} ({ensg})**", ""]

    # Subcellular localization
    main = rec.get("Subcellular main location") or []
    add = rec.get("Subcellular additional location") or []
    if main or add:
        out.append("**Subcellular location:**")
        if main:
            out.append(f"  Main: {', '.join(main)}")
        if add:
            out.append(f"  Additional: {', '.join(add)}")
        out.append("")

    # Single-cell type (which human cell types express it)
    scs = rec.get("RNA single cell type specificity")
    scsm = _fmt_ntpm(rec.get("RNA single cell type specific nCPM"))
    scd = rec.get("RNA single cell type distribution")
    if scs or scsm:
        out.append("**Single-cell type (human scRNA):**")
        if scs:
            out.append(f"  {scs}" + (f" — enriched: {scsm}" if scsm else ""))
        if scd:
            out.append(f"  Distribution: {scd}")
        out.append("")

    # Tissue specificity
    ts, td = rec.get("RNA tissue specificity"), rec.get("RNA tissue distribution")
    tsm = _fmt_ntpm(rec.get("RNA tissue specific nTPM"))
    if ts or td:
        out.append("**Tissue specificity (bulk RNA):**")
        if ts:
            out.append(f"  {ts}" + (f" — enriched: {tsm}" if tsm else ""))
        if td:
            out.append(f"  Distribution: {td}")
        out.append("")

    # Brain regional
    brs = rec.get("RNA brain regional specificity")
    brsm = _fmt_ntpm(rec.get("RNA brain regional specific nTPM"))
    if brs:
        out.append("**Brain regional expression:**")
        out.append(f"  {brs}" + (f" — {brsm}" if brsm else ""))
        out.append("")

    # Immune / blood + secretome
    bcs = rec.get("RNA blood cell specificity")
    bcsm = _fmt_ntpm(rec.get("RNA blood cell specific nTPM"))
    pc = rec.get("Protein class") or []
    secret = [p for p in pc if "secret" in p.lower() or "plasma protein" in p.lower()]
    if bcs or secret:
        out.append("**Immune / blood:**")
        if bcs:
            out.append(f"  Blood cell specificity: {bcs}" + (f" — {bcsm}" if bcsm else ""))
        if secret:
            out.append(f"  Secretome: {', '.join(secret)}")
        out.append("")

    # Cancer prognostics — show only the significant cohorts
    sig = []
    for k, v in rec.items():
        if k.startswith("Cancer prognostics") and isinstance(v, dict) and v.get("is_prognostic"):
            cancer = k.replace("Cancer prognostics - ", "").replace(" (TCGA)", "")
            ptype = v.get("prognostic type") or v.get("prognostic") or "prognostic"
            sig.append(f"  {cancer}: {ptype} (p={v.get('p_val')})")
    if sig:
        out.append("**Cancer prognostics (TCGA — significant cohorts):**")
        out.extend(sig)
    else:
        out.append("**Cancer prognostics:** not a significant prognostic marker in any of the 21 TCGA cohorts.")
    out.append("")

    if pc:
        out.append(f"**Protein class:** {', '.join(pc[:10])}")

    out.append("\nSource: Human Protein Atlas (proteinatlas.org)")
    return "\n".join(out)


if __name__ == "__main__":
    print(hpa_protein_atlas_tool.invoke({"gene_symbol": "EGFR"}))
