"""
CellxGene Census Single-Cell Expression Tool
=============================================
Efficient tool for querying gene expression at single-cell resolution
from the CZ CELLxGENE Census (60M+ cells).

Key efficiency strategies:
1. Filter genes server-side via var_value_filter (avoids loading all 60K genes)
2. Filter cells server-side via obs_value_filter (avoids loading all 60M cells)
3. Only fetch columns needed for the summary
4. Compute aggregated statistics instead of returning raw matrices
5. Use is_primary_data=True to avoid duplicate cells
"""

import logging
from typing import Optional
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Lazy import to avoid loading cellxgene_census until needed
_census_available = None

def _check_census_available():
    """Check if cellxgene_census is installed."""
    global _census_available
    if _census_available is None:
        try:
            import cellxgene_census
            _census_available = True
        except ImportError:
            _census_available = False
    return _census_available


def _resolve_gene_symbol(census, organism: str, gene_input: str) -> dict:
    """
    Resolve a gene symbol or Ensembl ID to its Census metadata.
    Returns dict with feature_id, feature_name, or None if not found.
    """
    import cellxgene_census
    
    gene_input = gene_input.strip()
    
    # Try as gene symbol first (feature_name)
    try:
        var_df = cellxgene_census.get_var(
            census, organism,
            value_filter=f"feature_name == '{gene_input}'",
            column_names=["feature_id", "feature_name", "feature_length"]
        )
        if len(var_df) > 0:
            row = var_df.iloc[0]
            return {
                "feature_id": row["feature_id"],
                "feature_name": row["feature_name"],
                "feature_length": row.get("feature_length", "N/A"),
                "matched_by": "symbol"
            }
    except Exception as e:
        logger.debug(f"Symbol lookup failed: {e}")
    
    # Try as Ensembl ID (feature_id)
    if gene_input.startswith("ENSG") or gene_input.startswith("ENSMUSG"):
        try:
            var_df = cellxgene_census.get_var(
                census, organism,
                value_filter=f"feature_id == '{gene_input}'",
                column_names=["feature_id", "feature_name", "feature_length"]
            )
            if len(var_df) > 0:
                row = var_df.iloc[0]
                return {
                    "feature_id": row["feature_id"],
                    "feature_name": row["feature_name"],
                    "feature_length": row.get("feature_length", "N/A"),
                    "matched_by": "ensembl_id"
                }
        except Exception as e:
            logger.debug(f"Ensembl ID lookup failed: {e}")
    
    # Try case-insensitive search (more expensive)
    try:
        # Get a broader set and filter locally
        var_df = cellxgene_census.get_var(
            census, organism,
            column_names=["feature_id", "feature_name", "feature_length"]
        )
        # Case-insensitive match
        mask = var_df["feature_name"].str.upper() == gene_input.upper()
        matches = var_df[mask]
        if len(matches) > 0:
            row = matches.iloc[0]
            return {
                "feature_id": row["feature_id"],
                "feature_name": row["feature_name"],
                "feature_length": row.get("feature_length", "N/A"),
                "matched_by": "case_insensitive"
            }
    except Exception as e:
        logger.debug(f"Case-insensitive lookup failed: {e}")
    
    return None


def _get_available_tissues(census, organism: str) -> list:
    """Get list of available tissue_general values."""
    import cellxgene_census
    
    obs_df = cellxgene_census.get_obs(
        census, organism,
        column_names=["tissue_general"],
        value_filter="is_primary_data == True"
    )
    return sorted(obs_df["tissue_general"].unique().tolist())


@tool
def singlecell_gene_expression_tool(
    gene: str,
    tissue: str,
    organism: str = "Homo sapiens",
    max_cell_types: int = 15
) -> str:
    """
    Query single-cell gene expression from CZ CELLxGENE Census (60M+ cells).
    
    Returns expression statistics BY CELL TYPE within a tissue, including:
    - Number of cells per cell type
    - Percentage of cells expressing the gene (count > 0)
    - Mean expression (raw counts) in expressing cells
    
    This provides CELL-TYPE RESOLUTION data, complementing bulk tissue data from GTEx.
    
    Args:
        gene: Gene symbol (e.g., "EGFR", "TP53") or Ensembl ID (e.g., "ENSG00000146648")
        tissue: Tissue to query (e.g., "lung", "brain", "heart", "blood"). 
                Use tissue_general categories.
        organism: "Homo sapiens" (default) or "Mus musculus"
        max_cell_types: Maximum number of cell types to report (default 15, sorted by cell count)
    
    Returns:
        Formatted string with single-cell expression statistics broken down by cell type.
    """
    import numpy as np
    
    output = []
    sources_status = {"CellxGene Census": "❌ NOT QUERIED"}
    
    # Check if census is available
    if not _check_census_available():
        output.append("❌ cellxgene_census package not installed.")
        output.append("Install with: pip install cellxgene-census")
        return "\n".join(output)
    
    import cellxgene_census
    
    try:
        with cellxgene_census.open_soma(census_version="2025-11-08") as census:
            # Step 1: Resolve gene symbol to Ensembl ID
            gene_info = _resolve_gene_symbol(census, organism.lower().replace(" ", "_"), gene)
            
            if gene_info is None:
                sources_status["CellxGene Census"] = "⚠️ GENE NOT FOUND"
                output.append(f"❌ Gene '{gene}' not found in Census for {organism}")
                output.append("\nTry:")
                output.append("  - Check spelling (case-sensitive for symbols)")
                output.append("  - Use official HGNC symbol")
                output.append("  - Try Ensembl ID (e.g., ENSG00000146648)")
                output.append("\n" + "="*50)
                output.append("**DATA SOURCE STATUS:**")
                output.append(f"  • CellxGene Census: {sources_status['CellxGene Census']}")
                return "\n".join(output)
            
            feature_id = gene_info["feature_id"]
            feature_name = gene_info["feature_name"]
            
            output.append(f"**Single-Cell Expression: {feature_name}** ({feature_id})")
            output.append(f"Organism: {organism} | Tissue: {tissue}")
            output.append("")
            
            # Step 2: Build efficient query filters
            # Key efficiency: filter BOTH genes AND cells server-side
            tissue_lower = tissue.lower().strip()
            
            obs_filter = f"tissue_general == '{tissue_lower}' and is_primary_data == True"
            var_filter = f"feature_id == '{feature_id}'"
            
            # Step 3: Fetch minimal AnnData slice
            # Only get cell_type column to minimize data transfer
            try:
                adata = cellxgene_census.get_anndata(
                    census,
                    organism=organism,
                    var_value_filter=var_filter,
                    obs_value_filter=obs_filter,
                    column_names={"obs": ["cell_type"], "var": ["feature_name"]}
                )
            except Exception as e:
                # Tissue might not exist - provide available options
                if "tissue_general" in str(e).lower() or "value_filter" in str(e).lower():
                    sources_status["CellxGene Census"] = "⚠️ TISSUE NOT FOUND"
                    output.append(f"❌ Tissue '{tissue}' not found or no cells match.")
                    try:
                        available = _get_available_tissues(
                            census, organism.lower().replace(" ", "_")
                        )
                        output.append(f"\nAvailable tissues: {', '.join(available[:20])}")
                        if len(available) > 20:
                            output.append(f"  ... and {len(available) - 20} more")
                    except:
                        pass
                else:
                    sources_status["CellxGene Census"] = f"❌ QUERY ERROR"
                    output.append(f"❌ Query failed: {str(e)[:100]}")
                
                output.append("\n" + "="*50)
                output.append("**DATA SOURCE STATUS:**")
                output.append(f"  • CellxGene Census: {sources_status['CellxGene Census']}")
                return "\n".join(output)
            
            n_cells = adata.n_obs
            
            if n_cells == 0:
                sources_status["CellxGene Census"] = "⚠️ NO CELLS FOUND"
                output.append(f"No cells found for {feature_name} in {tissue}")
                output.append("\n" + "="*50)
                output.append("**DATA SOURCE STATUS:**")
                output.append(f"  • CellxGene Census: {sources_status['CellxGene Census']}")
                return "\n".join(output)
            
            sources_status["CellxGene Census"] = "✅ SUCCESS"
            
            # Step 4: Compute per-cell-type statistics efficiently
            # Get expression vector (single gene, so it's 1D)
            X = adata.X
            if hasattr(X, 'toarray'):
                expr = X.toarray().flatten()
            else:
                expr = np.asarray(X).flatten()
            
            cell_types = adata.obs["cell_type"].values
            
            # Group by cell type and compute stats
            stats = []
            unique_types = np.unique(cell_types)
            
            for ct in unique_types:
                mask = cell_types == ct
                ct_expr = expr[mask]
                n_ct = len(ct_expr)
                n_expressing = np.sum(ct_expr > 0)
                pct_expressing = (n_expressing / n_ct) * 100 if n_ct > 0 else 0
                mean_expr = np.mean(ct_expr[ct_expr > 0]) if n_expressing > 0 else 0
                
                stats.append({
                    "cell_type": ct,
                    "n_cells": n_ct,
                    "n_expressing": n_expressing,
                    "pct_expressing": pct_expressing,
                    "mean_expr_in_expressing": mean_expr
                })
            
            # Sort by cell count (most populous first) and limit
            stats = sorted(stats, key=lambda x: x["n_cells"], reverse=True)
            
            # Summary statistics
            total_expressing = sum(s["n_expressing"] for s in stats)
            overall_pct = (total_expressing / n_cells) * 100
            
            output.append(f"**Summary:** {n_cells:,} cells | {len(unique_types)} cell types | {overall_pct:.1f}% expressing overall")
            output.append("")
            output.append("**Expression by Cell Type** (sorted by cell count):")
            output.append("-" * 70)
            output.append(f"{'Cell Type':<35} {'Cells':>8} {'% Expr':>8} {'Mean*':>8}")
            output.append("-" * 70)
            
            for s in stats[:max_cell_types]:
                ct_name = s["cell_type"][:34]
                output.append(
                    f"{ct_name:<35} {s['n_cells']:>8,} {s['pct_expressing']:>7.1f}% {s['mean_expr_in_expressing']:>8.1f}"
                )
            
            if len(stats) > max_cell_types:
                output.append(f"  ... and {len(stats) - max_cell_types} more cell types")
            
            output.append("-" * 70)
            output.append("*Mean = mean raw counts in expressing cells (count > 0)")
            output.append("")
            
            # Top expressing cell types
            top_expressing = sorted(stats, key=lambda x: x["pct_expressing"], reverse=True)[:5]
            if top_expressing:
                output.append("**Top expressing cell types:**")
                for s in top_expressing:
                    if s["pct_expressing"] > 0:
                        output.append(f"  • {s['cell_type']}: {s['pct_expressing']:.1f}% ({s['n_cells']:,} cells)")
    
    except Exception as e:
        sources_status["CellxGene Census"] = f"❌ ERROR: {str(e)[:50]}"
        output.append(f"❌ Census query error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Data source status footer
    output.append("\n" + "="*50)
    output.append("**DATA SOURCE STATUS:**")
    for source, status in sources_status.items():
        output.append(f"  • {source}: {status}")
    
    return "\n".join(output)


@tool
def singlecell_cell_types_tool(
    tissue: str,
    organism: str = "Homo sapiens",
    top_n: int = 25
) -> str:
    """
    List available cell types in a tissue from CellxGene Census.
    
    Use this to discover what cell types are available before querying 
    gene expression. Helpful for understanding tissue composition.
    
    Args:
        tissue: Tissue to query (e.g., "lung", "brain", "heart", "blood")
        organism: "Homo sapiens" (default) or "Mus musculus"
        top_n: Number of top cell types to show (by cell count)
    
    Returns:
        List of cell types with cell counts in the specified tissue.
    """
    output = []
    sources_status = {"CellxGene Census": "❌ NOT QUERIED"}
    
    if not _check_census_available():
        output.append("❌ cellxgene_census package not installed.")
        return "\n".join(output)
    
    import cellxgene_census
    
    try:
        with cellxgene_census.open_soma(census_version="2025-11-08") as census:
            tissue_lower = tissue.lower().strip()
            
            obs_df = cellxgene_census.get_obs(
                census,
                organism.lower().replace(" ", "_"),
                value_filter=f"tissue_general == '{tissue_lower}' and is_primary_data == True",
                column_names=["cell_type"]
            )
            
            if len(obs_df) == 0:
                sources_status["CellxGene Census"] = "⚠️ NO DATA"
                output.append(f"No cells found for tissue '{tissue}'")
                # List available tissues
                try:
                    available = _get_available_tissues(
                        census, organism.lower().replace(" ", "_")
                    )
                    output.append(f"\nAvailable tissues: {', '.join(available)}")
                except:
                    pass
            else:
                sources_status["CellxGene Census"] = "✅ SUCCESS"
                
                counts = obs_df["cell_type"].value_counts()
                total = len(obs_df)
                
                output.append(f"**Cell Types in {tissue.title()}** ({organism})")
                output.append(f"Total: {total:,} cells | {len(counts)} cell types")
                output.append("")
                output.append(f"{'Cell Type':<40} {'Count':>10} {'%':>7}")
                output.append("-" * 60)
                
                for ct, count in counts.head(top_n).items():
                    pct = (count / total) * 100
                    output.append(f"{ct[:39]:<40} {count:>10,} {pct:>6.1f}%")
                
                if len(counts) > top_n:
                    output.append(f"  ... and {len(counts) - top_n} more cell types")
    
    except Exception as e:
        sources_status["CellxGene Census"] = f"❌ ERROR"
        output.append(f"Error: {str(e)}")
    
    output.append("\n" + "="*50)
    output.append("**DATA SOURCE STATUS:**")
    output.append(f"  • CellxGene Census: {sources_status['CellxGene Census']}")
    
    return "\n".join(output)


# For testing
if __name__ == "__main__":
    # Test the tool
    print("Testing singlecell_gene_expression_tool...")
    result = singlecell_gene_expression_tool.invoke({
        "gene": "EGFR",
        "tissue": "lung",
        "organism": "Homo sapiens"
    })
    print(result)
    print("\n" + "="*80 + "\n")
    
    print("Testing singlecell_cell_types_tool...")
    result = singlecell_cell_types_tool.invoke({
        "tissue": "brain",
        "organism": "Homo sapiens"
    })
    print(result)
