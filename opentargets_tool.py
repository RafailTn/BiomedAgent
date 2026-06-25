"""
Open Targets gene–disease association tool.
===========================================
Replacement for the PrimeKG knowledge-graph lookups: instead of a 2.3 GB local
graph + GLiNER, this queries the Open Targets Platform GraphQL API live for
target (gene) ↔ disease associations with evidence-backed association scores.

No API key required. Uses the shared cached/retrying HTTP session.

API: https://platform.opentargets.org/api  (GraphQL v4)
"""

import logging
from typing import Optional

import http_client

logger = logging.getLogger(__name__)

GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"

# 1) Resolve a gene symbol to its Ensembl target id.
_SEARCH_QUERY = """
query ($q: String!) {
  search(queryString: $q, entityNames: ["target"], page: {index: 0, size: 5}) {
    hits { id entity name }
  }
}
"""

# 2) Top diseases associated with that target, by overall association score.
_ASSOC_QUERY = """
query ($id: String!, $size: Int!) {
  target(ensemblId: $id) {
    approvedSymbol
    associatedDiseases(page: {index: 0, size: $size}) {
      count
      rows {
        score
        disease { id name }
      }
    }
  }
}
"""


def _graphql(query: str, variables: dict) -> Optional[dict]:
    try:
        r = http_client.post(GRAPHQL_URL, json={"query": query, "variables": variables})
        if r.ok:
            data = r.json()
            if data.get("errors"):
                logger.warning("Open Targets GraphQL errors: %s", data["errors"])
            return data.get("data")
        logger.warning("Open Targets HTTP %s", r.status_code)
    except Exception as e:
        logger.warning("Open Targets request failed: %s", e)
    return None


def _resolve_target(gene_symbol: str) -> Optional[dict]:
    """Return {'id': ENSG..., 'name': symbol} for the best target hit, or None."""
    data = _graphql(_SEARCH_QUERY, {"q": gene_symbol})
    if not data:
        return None
    hits = (data.get("search") or {}).get("hits") or []
    # Prefer an exact (case-insensitive) symbol match, else the first target hit.
    for h in hits:
        if h.get("entity") == "target" and (h.get("name", "").upper() == gene_symbol.upper()):
            return h
    for h in hits:
        if h.get("entity") == "target":
            return h
    return None


def query_gene_diseases(gene_symbol: str, size: int = 15) -> str:
    """Format the top Open Targets disease associations for a gene as plain text."""
    gene_symbol = gene_symbol.strip()
    out = [f"**Open Targets — Diseases associated with {gene_symbol.upper()}**"]
    status = "❌ NOT QUERIED"

    target = _resolve_target(gene_symbol)
    if not target:
        out.append("")
        out.append(f"⚠️ No Open Targets target found for '{gene_symbol}'.")
        out.append("**DATA SOURCE STATUS:** Open Targets: ⚠️ TARGET NOT FOUND")
        out.append("DO NOT invent associations.")
        return "\n".join(out)

    ensembl_id = target["id"]
    data = _graphql(_ASSOC_QUERY, {"id": ensembl_id, "size": size})

    target_node = (data or {}).get("target") or {}
    assoc = target_node.get("associatedDiseases") or {}
    rows = assoc.get("rows") or []
    symbol = target_node.get("approvedSymbol", gene_symbol.upper())

    if not rows:
        out.append(f"  Ensembl target: {ensembl_id}")
        out.append("")
        out.append("No disease associations returned.")
        out.append("**DATA SOURCE STATUS:** Open Targets: ⚠️ NO ASSOCIATIONS")
        return "\n".join(out)

    status = "✅ SUCCESS"
    total = assoc.get("count", len(rows))
    out.append(f"  Approved symbol: {symbol}  |  Ensembl: {ensembl_id}")
    out.append(f"  Showing top {len(rows)} of {total:,} associations "
               f"(overall association score, 0–1):")
    out.append("")
    for i, row in enumerate(rows, 1):
        d = row.get("disease") or {}
        out.append(f"  {i:>2}. {d.get('name', 'Unknown'):<45} "
                   f"score={row.get('score', 0):.3f}  [{d.get('id', 'N/A')}]")

    out.append("")
    out.append("Score = Open Targets overall association strength (genetics, drugs, "
               "literature, expression, etc.). Higher = stronger evidence.")
    out.append(f"**DATA SOURCE STATUS:** Open Targets: {status}")
    return "\n".join(out)


if __name__ == "__main__":
    print(query_gene_diseases("TP53"))
