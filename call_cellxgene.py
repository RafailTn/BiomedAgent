"""
Single-Cell Gene Expression Query Tool - FIXED VERSION
Uses CZ CELLxGENE Census API for fast, programmatic access to single-cell RNA-seq data.
Returns interpretable results suitable for AI agent consumption.
"""

import json
import sys
import warnings
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Union, Any
import cellxgene_census
import pandas as pd
import numpy as np
from scipy import sparse

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


@dataclass
class GeneExpressionResult:
    """Structured result for AI agent interpretation."""
    query_id: str
    organism: str
    cell_count: int
    gene_count: int
    cell_types: List[Dict[str, Any]]
    tissue_summary: List[Dict[str, Any]]
    expression_summary: Dict[str, Any]
    metadata: Dict[str, Any]
    query_params: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SingleCellQueryTool:
    """
    Tool for querying single-cell gene expression data from CELLxGENE Census.
    Optimized for AI agent integration with fast, structured responses.
    """
    
    def __init__(self, census_version: Optional[str] = "2025-11-08"):
        """
        Initialize the query tool.
        
        Args:
            census_version: Specific version to use (e.g., "2023-12-15"). 
                          If None, uses latest stable version.
        """
        self.census_version = census_version
        self.census = None
        
    def __enter__(self):
        """Context manager entry."""
        self.census = cellxgene_census.open_soma(census_version=self.census_version)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.census:
            self.census.close()
            
    def explore_available_data(self, organism: str = "Homo sapiens") -> Dict:
        """
        Explore what data is available in the Census.
        Useful for AI agent to understand the data landscape.
        """
        try:
            # Get summary stats
            summary = self.census["census_info"]["summary"].read().concat().to_pandas()
            
            # Get datasets info
            datasets = self.census["census_info"]["datasets"].read().concat().to_pandas()
            
            # Get unique cell types and tissues (sample for speed)
            obs = self.census["census_data"][organism.lower().replace(" ", "_")].obs
            cell_types = obs.read(column_names=["cell_type"]).concat().to_pandas()
            tissues = obs.read(column_names=["tissue_general"]).concat().to_pandas()
            
            return {
                "total_cells": int(summary["total_cell_count"].iloc[0]) if not summary.empty else 0,
                "total_datasets": len(datasets),
                "unique_cell_types": cell_types["cell_type"].nunique() if not cell_types.empty else 0,
                "unique_tissues": tissues["tissue_general"].nunique() if not tissues.empty else 0,
                "sample_cell_types": cell_types["cell_type"].value_counts().head(10).to_dict() if not cell_types.empty else {},
                "sample_tissues": tissues["tissue_general"].value_counts().head(10).to_dict() if not tissues.empty else {},
                "disease_coverage": datasets["disease"].value_counts().head(10).to_dict() if "disease" in datasets.columns and not datasets.empty else {}
            }
        except Exception as e:
            return {"error": str(e), "total_cells": 0, "total_datasets": 0}
    
    def query_expression(
        self,
        organism: str = "Homo sapiens",
        genes: Optional[List[str]] = None,
        gene_ids: Optional[List[str]] = None,
        cell_type: Optional[str] = None,
        tissue: Optional[str] = None,
        disease: Optional[str] = None,
        sex: Optional[str] = None,
        development_stage: Optional[str] = None,
        assay: Optional[str] = None,
        max_cells: Optional[int] = None,
        include_primary_only: bool = True
    ) -> GeneExpressionResult:
        """
        Query gene expression data with flexible filters.
        
        Args:
            organism: "Homo sapiens" or "Mus musculus"
            genes: List of gene symbols (e.g., ["CD4", "CD8A", "FOXP3"])
            gene_ids: List of Ensembl IDs (e.g., ["ENSG00000161798"])
            cell_type: Specific cell type (e.g., "B cell", "T cell")
            tissue: Tissue name (e.g., "lung", "blood")
            disease: Disease state (e.g., "COVID-19", "normal")
            sex: "male", "female", or "unknown"
            development_stage: Developmental stage
            assay: Sequencing assay type
            max_cells: Maximum cells to return (None for all)
            include_primary_only: Exclude duplicate cells across datasets
            
        Returns:
            GeneExpressionResult with structured data for AI interpretation
        """
        warnings_list = []
        query_params = {
            "organism": organism,
            "genes": genes,
            "gene_ids": gene_ids,
            "cell_type": cell_type,
            "tissue": tissue,
            "disease": disease,
            "sex": sex,
            "development_stage": development_stage,
            "assay": assay,
            "max_cells": max_cells,
            "include_primary_only": include_primary_only
        }
        
        # Build filter strings
        obs_filters = []
        
        if cell_type:
            obs_filters.append(f"cell_type == '{cell_type}'")
        if tissue:
            obs_filters.append(f"tissue_general == '{tissue}'")
        if disease:
            obs_filters.append(f"disease == '{disease}'")
        if sex:
            obs_filters.append(f"sex == '{sex}'")
        if development_stage:
            obs_filters.append(f"development_stage == '{development_stage}'")
        if assay:
            obs_filters.append(f"assay == '{assay}'")
        if include_primary_only:
            obs_filters.append("is_primary_data == True")
            
        obs_value_filter = " and ".join(obs_filters) if obs_filters else None
        
        # Build gene filter
        var_value_filter = None
        if gene_ids:
            id_list = "', '".join(gene_ids)
            var_value_filter = f"feature_id in ['{id_list}']"
        elif genes:
            gene_list = "', '".join(genes)
            var_value_filter = f"feature_name in ['{gene_list}']"
        
        # Query the data
        print(f"Querying Census: organism={organism}, cells matching={obs_value_filter or 'all'}, genes={var_value_filter or 'all'}")
        
        try:
            adata = cellxgene_census.get_anndata(
                census=self.census,
                organism=organism,
                obs_value_filter=obs_value_filter,
                var_value_filter=var_value_filter,
                obs_column_names=["cell_type", "tissue_general", "tissue", "disease", 
                                "sex", "donor_id", "assay", "development_stage", 
                                "dataset_id", "is_primary_data"],
                var_column_names=["feature_id", "feature_name", "feature_length"]
            )
        except Exception as e:
            warnings_list.append(f"Query failed: {str(e)}")
            return self._create_empty_result(organism, query_params, warnings_list)
        
        # Check if we got any data
        if adata.n_obs == 0:
            warnings_list.append("No cells found matching the specified criteria")
            return self._create_empty_result(organism, query_params, warnings_list)
        
        if adata.n_vars == 0:
            warnings_list.append("No genes found matching the specified criteria")
            return self._create_empty_result(organism, query_params, warnings_list)
        
        # Apply cell limit if specified
        if max_cells and adata.n_obs > max_cells:
            print(f"Subsampling from {adata.n_obs} to {max_cells} cells")
            try:
                import scanpy as sc
                sc.pp.subsample(adata, n_obs=max_cells, random_state=42, copy=False)
            except ImportError:
                warnings_list.append("scanpy not installed, using random subsampling")
                indices = np.random.choice(adata.n_obs, size=max_cells, replace=False)
                adata = adata[indices]
        
        # Generate interpretable summary
        result = self._summarize_expression_data(adata, organism, query_params, warnings_list)
        return result
    
    def _create_empty_result(self, organism: str, query_params: Dict, warnings: List[str]) -> GeneExpressionResult:
        """Create an empty result when no data is found."""
        return GeneExpressionResult(
            query_id=f"sc_query_empty_{hash(str(query_params))}",
            organism=organism,
            cell_count=0,
            gene_count=0,
            cell_types=[],
            tissue_summary=[],
            expression_summary={
                "genes_analyzed": [],
                "expression_stats": {},
                "total_umi_counts": 0.0,
                "mean_reads_per_cell": 0.0
            },
            metadata={
                "datasets": {},
                "diseases": {},
                "assays": {},
                "sex_distribution": {},
                "donor_count": 0
            },
            query_params=query_params,
            warnings=warnings
        )
    
    def _summarize_expression_data(self, adata, organism: str, query_params: Dict, warnings: List[str]) -> GeneExpressionResult:
        """
        Create AI-friendly summary of expression data.
        FIXED: Handles division by zero and empty data gracefully.
        """
        # Cell type breakdown - SAFE
        cell_type_summary = []
        if "cell_type" in adata.obs.columns and adata.n_obs > 0:
            cell_type_counts = adata.obs["cell_type"].value_counts()
            for ct, count in cell_type_counts.head(20).items():
                cell_type_summary.append({
                    "cell_type": ct, 
                    "count": int(count), 
                    "percentage": round(count/max(adata.n_obs, 1)*100, 2)  # SAFE: max with 1
                })
        
        # Tissue breakdown - SAFE
        tissue_summary = []
        if "tissue_general" in adata.obs.columns and adata.n_obs > 0:
            tissue_counts = adata.obs["tissue_general"].value_counts()
            for tissue, count in tissue_counts.head(20).items():
                tissue_summary.append({
                    "tissue": tissue, 
                    "count": int(count), 
                    "percentage": round(count/max(adata.n_obs, 1)*100, 2)  # SAFE: max with 1
                })
        
        # Expression statistics per gene - SAFE
        expression_stats = {}
        total_umi = 0.0
        mean_reads = 0.0
        
        if adata.n_vars > 0 and adata.n_obs > 0:
            try:
                # Calculate total UMI safely
                if sparse.issparse(adata.X):
                    total_umi = float(adata.X.sum())
                else:
                    total_umi = float(np.sum(adata.X))
                
                # Mean reads per cell - SAFE
                mean_reads = total_umi / max(adata.n_obs, 1)
                
                # Convert to dense for summary stats if small enough, otherwise use sparse ops
                matrix_size = adata.n_obs * adata.n_vars
                
                if matrix_size < 1e5:
                    expr_matrix = adata.X.toarray() if sparse.issparse(adata.X) else adata.X
                    
                    for i, gene in enumerate(adata.var["feature_name"]):
                        gene_expr = expr_matrix[:, i]
                        non_zero_count = np.sum(gene_expr > 0)
                        
                        expression_stats[gene] = {
                            "mean_expression": float(np.mean(gene_expr)),
                            "median_expression": float(np.median(gene_expr)),
                            "pct_expressing": float(non_zero_count / max(len(gene_expr), 1) * 100),  # SAFE
                            "max_expression": float(np.max(gene_expr)),
                            "variance": float(np.var(gene_expr)),
                            "expressing_cells": int(non_zero_count)
                        }
                else:
                    # For large matrices, compute sparse statistics
                    for i, gene in enumerate(adata.var["feature_name"]):
                        if sparse.issparse(adata.X):
                            col = adata.X[:, i].toarray().flatten()
                        else:
                            col = adata.X[:, i]
                        
                        non_zero_count = np.sum(col > 0)
                        expression_stats[gene] = {
                            "mean_expression": float(np.mean(col)),
                            "pct_expressing": float(non_zero_count / max(len(col), 1) * 100),  # SAFE
                            "expressing_cells": int(non_zero_count)
                        }
                        
            except Exception as e:
                warnings.append(f"Error computing expression statistics: {str(e)}")
        
        # Dataset provenance - SAFE
        dataset_summary = {}
        if "dataset_id" in adata.obs.columns and not adata.obs.empty:
            dataset_summary = adata.obs["dataset_id"].value_counts().head(10).to_dict()
            dataset_summary = {k: int(v) for k, v in dataset_summary.items()}
        
        # Disease summary - SAFE
        disease_summary = {}
        if "disease" in adata.obs.columns and not adata.obs.empty:
            disease_summary = adata.obs["disease"].value_counts().head(10).to_dict()
            disease_summary = {k: int(v) for k, v in disease_summary.items()}
        
        # Assays - SAFE
        assays = {}
        if "assay" in adata.obs.columns and not adata.obs.empty:
            assays = adata.obs["assay"].value_counts().to_dict()
            assays = {k: int(v) for k, v in assays.items()}
        
        # Sex distribution - SAFE
        sex_dist = {}
        if "sex" in adata.obs.columns and not adata.obs.empty:
            sex_dist = adata.obs["sex"].value_counts().to_dict()
            sex_dist = {k: int(v) for k, v in sex_dist.items()}
        
        # Donor count - SAFE
        donor_count = 0
        if "donor_id" in adata.obs.columns and not adata.obs.empty:
            donor_count = int(adata.obs["donor_id"].nunique())
        
        return GeneExpressionResult(
            query_id=f"sc_query_{hash(str(adata.obs.head()))}",
            organism=organism,
            cell_count=int(adata.n_obs),
            gene_count=int(adata.n_vars),
            cell_types=cell_type_summary,
            tissue_summary=tissue_summary,
            expression_summary={
                "genes_analyzed": list(expression_stats.keys()),
                "expression_stats": expression_stats,
                "total_umi_counts": total_umi,
                "mean_reads_per_cell": round(mean_reads, 2)
            },
            metadata={
                "datasets": dataset_summary,
                "diseases": disease_summary,
                "assays": assays,
                "sex_distribution": sex_dist,
                "donor_count": donor_count
            },
            query_params=query_params,
            warnings=warnings
        )
    
    def find_marker_genes(
        self, 
        organism: str = "Homo sapiens",
        cell_type: str = "T cell",
        tissue: str = "blood",
        top_n: int = 10
    ) -> Dict:
        """
        Simple marker gene identification by differential expression vs all other cells.
        Returns top marker genes for the specified cell type.
        """
        warnings_list = []
        
        try:
            # Query target cells
            adata_target = cellxgene_census.get_anndata(
                census=self.census,
                organism=organism,
                obs_value_filter=f"cell_type == '{cell_type}' and tissue_general == '{tissue}' and is_primary_data == True",
                obs_column_names=["cell_type"],
                var_column_names=["feature_name"]
            )
            
            if adata_target.n_obs == 0:
                return {
                    "error": f"No cells found for {cell_type} in {tissue}",
                    "cell_type": cell_type,
                    "tissue": tissue,
                    "warnings": warnings_list
                }
            
            # Query other cells (sample for comparison)
            adata_other = cellxgene_census.get_anndata(
                census=self.census,
                organism=organism,
                obs_value_filter=f"cell_type != '{cell_type}' and tissue_general == '{tissue}' and is_primary_data == True",
                obs_column_names=["cell_type"],
                var_column_names=["feature_name"]
            )
            
            if adata_other.n_obs == 0:
                return {
                    "error": f"No comparison cells found in {tissue}",
                    "cell_type": cell_type,
                    "tissue": tissue,
                    "warnings": warnings_list
                }
            
            if adata_other.n_obs > 10000:
                try:
                    import scanpy as sc
                    sc.pp.subsample(adata_other, n_obs=10000, random_state=42, copy=False)
                except ImportError:
                    warnings_list.append("scanpy not installed, using random subsampling")
                    indices = np.random.choice(adata_other.n_obs, size=10000, replace=False)
                    adata_other = adata_other[indices]
            
            # Simple differential expression (mean log2 fold change)
            target_means = np.array(adata_target.X.mean(axis=0)).flatten()
            other_means = np.array(adata_other.X.mean(axis=0)).flatten()
            
            # SAFE: Add pseudocount to avoid log(0) and division by zero
            target_means_safe = target_means + 1e-6
            other_means_safe = other_means + 1e-6
            
            log2_fc = np.log2(target_means_safe / other_means_safe)
            
            # Get top genes
            gene_names = adata_target.var["feature_name"].values
            top_indices = np.argsort(log2_fc)[-top_n:][::-1]
            
            markers = []
            for idx in top_indices:
                markers.append({
                    "gene": gene_names[idx],
                    "log2_fold_change": float(log2_fc[idx]),
                    "mean_expression_target": float(target_means[idx]),
                    "mean_expression_other": float(other_means[idx])
                })
                
            return {
                "cell_type": cell_type,
                "tissue": tissue,
                "target_cells": int(adata_target.n_obs),
                "comparison_cells": int(adata_other.n_obs),
                "marker_genes": markers,
                "warnings": warnings_list
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "cell_type": cell_type,
                "tissue": tissue,
                "warnings": warnings_list
            }


def main():
    """
    CLI interface for the single-cell query tool.
    Example usage for AI agent integration.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Query single-cell gene expression data")
    parser.add_argument("--cell-type", type=str, help="Cell type to query (e.g., 'B cell')")
    parser.add_argument("--tissue", type=str, help="Tissue (e.g., 'lung', 'blood')")
    parser.add_argument("--disease", type=str, help="Disease state")
    parser.add_argument("--genes", type=str, help="Comma-separated gene symbols")
    parser.add_argument("--organism", type=str, default="Homo sapiens", 
                       choices=["Homo sapiens", "Mus musculus"])
    parser.add_argument("--max-cells", type=int, default=10000, 
                       help="Maximum cells to retrieve")
    parser.add_argument("--output", type=str, default="sc_expression_result.json",
                       help="Output JSON file")
    parser.add_argument("--explore", action="store_true", 
                       help="Explore available data types first")
    parser.add_argument("--find-markers", action="store_true",
                       help="Find marker genes for cell type")
    
    args = parser.parse_args()
    
    # Initialize and run query
    with SingleCellQueryTool() as tool:
        if args.explore:
            # Exploration mode - show what's available
            print("Exploring available data in CELLxGENE Census...")
            available = tool.explore_available_data(args.organism)
            print(json.dumps(available, indent=2))
            return
            
        if args.find_markers and args.cell_type:
            # Marker gene discovery mode
            print(f"Finding marker genes for {args.cell_type} in {args.tissue or 'all tissues'}...")
            markers = tool.find_marker_genes(
                organism=args.organism,
                cell_type=args.cell_type,
                tissue=args.tissue or "blood"
            )
            print(json.dumps(markers, indent=2))
            with open(args.output, 'w') as f:
                json.dump(markers, f, indent=2)
            print(f"Saved to {args.output}")
            return
        
        # Standard expression query
        gene_list = args.genes.split(",") if args.genes else None
        
        print(f"Querying {args.organism} data...")
        result = tool.query_expression(
            organism=args.organism,
            genes=gene_list,
            cell_type=args.cell_type,
            tissue=args.tissue,
            disease=args.disease,
            max_cells=args.max_cells
        )
        
        # Output structured result
        result_dict = result.to_dict()
        print(json.dumps(result_dict, indent=2))
        
        # Save to file
        with open(args.output, 'w') as f:
            json.dump(result_dict, f, indent=2)
        print(f"\nSaved structured results to {args.output}")
        print(f"Total cells retrieved: {result.cell_count}")
        print(f"Genes analyzed: {result.gene_count}")
        
        if result.warnings:
            print(f"\nWarnings ({len(result.warnings)}):")
            for w in result.warnings:
                print(f"  - {w}")


if __name__ == "__main__":
    main()
