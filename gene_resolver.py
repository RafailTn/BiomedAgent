"""
MyGene.info gene resolver.
==========================
One call returns official symbol, name, aliases, summary, Entrez/Ensembl/UniProt
IDs and genomic position — and robustly resolves aliases (e.g. "p53" → "TP53").

Used as:
  * the primary backend for gene_info_tool, and
  * a shared symbol normalizer for the expression tools (GTEx, AtlasApprox).
"""

import logging
from typing import Optional

import http_client

logger = logging.getLogger(__name__)

MYGENE_URL = "https://mygene.info/v3"
_FIELDS = ("symbol,name,entrezgene,ensembl.gene,alias,summary,"
           "uniprot.Swiss-Prot,genomic_pos,type_of_gene")


def _aliases(hit: dict) -> list:
    a = hit.get("alias") or []
    return [a.lower()] if isinstance(a, str) else [str(x).lower() for x in a]


def _best_hit(hits: list, query: str) -> Optional[dict]:
    """Pick the canonical gene from MyGene hits, not just the top free-text match.

    Free-text relevance can rank a gene whose *summary* mentions the query above
    the gene the query actually names (e.g. 'HER2' -> PTPN18 because PTPN18
    dephosphorylates HER2). Prefer exact symbol, then an exact alias on a
    protein-coding gene.
    """
    if not hits:
        return None
    ql = query.lower()
    for h in hits:                                   # exact official symbol
        if h.get("symbol", "").lower() == ql:
            return h
    coding = [h for h in hits if h.get("type_of_gene") == "protein-coding"]
    for h in coding:                                 # exact alias, protein-coding
        if ql in _aliases(h):
            return h
    for h in hits:                                   # exact alias, any
        if ql in _aliases(h):
            return h
    return coding[0] if coding else hits[0]


def resolve_gene(query: str, species: str = "human") -> Optional[dict]:
    """Return the best MyGene.info hit for a symbol / alias / id, or None."""
    query = (query or "").strip()
    if not query:
        return None
    # Scope to symbol/alias first (precise), fall back to free text.
    for q in (f"symbol:{query} OR alias:{query}", query):
        try:
            r = http_client.get(
                f"{MYGENE_URL}/query",
                params={"q": q, "species": species, "fields": _FIELDS, "size": 10},
            )
            if r.ok:
                best = _best_hit(r.json().get("hits", []), query)
                if best:
                    return best
            else:
                logger.warning("MyGene.info HTTP %s for %r", r.status_code, query)
        except Exception as e:
            logger.warning("MyGene.info resolve failed for %r: %s", query, e)
    return None


def resolve_symbol(query: str, species: str = "human") -> Optional[str]:
    """Return the official gene symbol for a symbol/alias, or None if unresolved."""
    g = resolve_gene(query, species)
    return g.get("symbol") if g else None


if __name__ == "__main__":
    for q in ["p53", "HER2", "TP53", "notagene"]:
        g = resolve_gene(q)
        print(f"{q:10} -> {g.get('symbol') if g else None}")
