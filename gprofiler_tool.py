"""
g:Profiler functional enrichment (g:GOSt).
==========================================
Statistically-correct over-representation analysis for a gene SET across GO,
KEGG, Reactome, and WikiPathways in a single call (g:SCS multiple-testing
correction). Replaces the old naive KEGG overlap-counting tool.

API: https://biit.cs.ut.ee/gprofiler/  (no key required)
"""

import logging

from langchain_core.tools import tool

import http_client

logger = logging.getLogger(__name__)

GPROFILER_URL = "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"

ORGANISMS = {
    "human": "hsapiens", "homo sapiens": "hsapiens", "hsapiens": "hsapiens",
    "mouse": "mmusculus", "mus musculus": "mmusculus",
    "rat": "rnorvegicus", "zebrafish": "drerio", "fly": "dmelanogaster",
    "worm": "celegans", "yeast": "scerevisiae",
}

# Display order — pathways first (usually what users want), then GO.
_SRC_ORDER = ["KEGG", "REAC", "WP", "GO:BP", "GO:MF", "GO:CC"]
_SRC_LABEL = {"KEGG": "KEGG", "REAC": "Reactome", "WP": "WikiPathways",
              "GO:BP": "GO Biological Process", "GO:MF": "GO Molecular Function",
              "GO:CC": "GO Cellular Component"}


def run_enrichment(genes: list, organism: str = "human") -> list | None:
    """Return g:Profiler result rows, or None on request failure."""
    org = ORGANISMS.get(organism.lower(), organism.lower())
    payload = {
        "organism": org,
        "query": genes,
        "sources": ["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC", "WP"],
        "user_threshold": 0.05,
        "significance_threshold_method": "g_SCS",
        "no_evidences": True,
        "ordered": False,
    }
    try:
        r = http_client.post(GPROFILER_URL, json=payload)
        if r.ok:
            return r.json().get("result", [])
        logger.warning("g:Profiler HTTP %s", r.status_code)
    except Exception as e:
        logger.warning("g:Profiler request failed: %s", e)
    return None


@tool
def gprofiler_enrichment_tool(genes: str, organism: str = "human") -> str:
    """Functional enrichment (GO + KEGG + Reactome + WikiPathways, FDR-corrected) for a gene SET."""
    gene_list = [g.strip() for g in genes.replace(";", ",").split(",") if g.strip()]
    if not gene_list:
        return "No genes provided. Pass a comma-separated gene list (e.g. 'TP53,BRCA1,ATM')."

    results = run_enrichment(gene_list, organism)
    if results is None:
        return "❌ g:Profiler request failed."
    if not results:
        return f"No statistically significant enrichment for: {', '.join(gene_list)}"

    results.sort(key=lambda x: x.get("p_value", 1.0))
    by_src: dict = {}
    for t in results:
        by_src.setdefault(t.get("source", "?"), []).append(t)

    lines = [
        f"**g:Profiler Functional Enrichment** ({len(gene_list)} genes: {', '.join(gene_list)})",
        "FDR-corrected (g:SCS), adjusted p<0.05. Top terms per source:",
        "",
    ]
    for src in _SRC_ORDER:
        terms = by_src.get(src)
        if not terms:
            continue
        lines.append(f"**{_SRC_LABEL.get(src, src)}:**")
        for t in terms[:6]:
            lines.append(
                f"  • {t.get('name', '?')} "
                f"(adj p={t.get('p_value', 1):.1e}, "
                f"{t.get('intersection_size', 0)}/{t.get('term_size', 0)} genes) "
                f"[{t.get('native', '')}]"
            )
        lines.append("")

    lines.append("p = g:SCS-adjusted significance; ratio = your genes in term / total genes in term.")
    return "\n".join(lines)


if __name__ == "__main__":
    print(gprofiler_enrichment_tool.invoke({"genes": "TP53,BRCA1,ATM,CHEK2,BRCA2"}))
