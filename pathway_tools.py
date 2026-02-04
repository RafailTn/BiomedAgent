"""
Pathway & Protein Interaction Tools
====================================

Tools for querying:
1. STRING - Protein-protein interaction database
2. KEGG - Pathway database

Both use official REST APIs - no API key required.
"""

import requests
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ============================================================================
# STRING DATABASE (Protein-Protein Interactions)
# ============================================================================

class STRINGClient:
    """
    Client for STRING database API.
    
    STRING: Search Tool for the Retrieval of Interacting Genes/Proteins
    https://string-db.org/
    
    API Documentation: https://string-db.org/help/api/
    """
    
    BASE_URL = "https://string-db.org/api"
    
    # Common species taxonomy IDs
    SPECIES = {
        "human": 9606,
        "mouse": 10090,
        "rat": 10116,
        "zebrafish": 7955,
        "fly": 7227,
        "worm": 6239,
        "yeast": 4932,
        "ecoli": 511145,
    }
    
    def __init__(self, output_format: str = "json"):
        """
        Initialize STRING client.
        
        Args:
            output_format: Response format ("json", "tsv", "tsv-no-header", "psi-mi", "psi-mi-tab")
        """
        self.format = output_format
    
    def _request(self, endpoint: str, params: Dict) -> Any:
        """Make API request."""
        url = f"{self.BASE_URL}/{self.format}/{endpoint}"
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            if self.format == "json":
                return response.json()
            else:
                return response.text
                
        except requests.RequestException as e:
            logger.error(f"STRING API error: {e}")
            return {"error": str(e)}
    
    def get_interactions(
        self,
        proteins: List[str],
        species: str = "human",
        required_score: int = 400,
        network_type: str = "functional",
        limit: int = 50,
    ) -> Dict[str, Any]:
        """
        Get protein-protein interactions from STRING.
        
        Args:
            proteins: List of protein/gene names (e.g., ["TP53", "MDM2"])
            species: Species name or NCBI taxonomy ID
            required_score: Minimum combined score (0-1000). 
                           400=medium, 700=high, 900=highest confidence
            network_type: "functional" (all associations) or "physical" (direct binding only)
            limit: Maximum number of interactions to return
            
        Returns:
            Dict with interaction data including scores and evidence types
        """
        # Resolve species
        if isinstance(species, str) and species.lower() in self.SPECIES:
            species_id = self.SPECIES[species.lower()]
        else:
            species_id = int(species) if str(species).isdigit() else 9606
        
        params = {
            "identifiers": "%0d".join(proteins),  # Newline-separated
            "species": species_id,
            "required_score": required_score,
            "network_type": network_type,
            "limit": limit,
        }
        
        result = self._request("network", params)
        
        if isinstance(result, list):
            return {
                "query_proteins": proteins,
                "species": species_id,
                "min_score": required_score,
                "network_type": network_type,
                "n_interactions": len(result),
                "interactions": result,
            }
        return result
    
    def get_interaction_partners(
        self,
        protein: str,
        species: str = "human",
        required_score: int = 400,
        limit: int = 20,
    ) -> Dict[str, Any]:
        """
        Get interaction partners for a single protein.
        
        Args:
            protein: Protein/gene name (e.g., "TP53")
            species: Species name
            required_score: Minimum confidence score
            limit: Max partners to return
            
        Returns:
            Dict with interacting proteins and scores
        """
        if isinstance(species, str) and species.lower() in self.SPECIES:
            species_id = self.SPECIES[species.lower()]
        else:
            species_id = 9606
        
        params = {
            "identifiers": protein,
            "species": species_id,
            "required_score": required_score,
            "limit": limit,
        }
        
        result = self._request("interaction_partners", params)
        
        if isinstance(result, list):
            # Extract partner names and scores
            partners = []
            for interaction in result:
                # Determine which protein is the partner
                if interaction.get("preferredName_A", "").upper() == protein.upper():
                    partner_name = interaction.get("preferredName_B", "")
                else:
                    partner_name = interaction.get("preferredName_A", "")
                
                partners.append({
                    "partner": partner_name,
                    "combined_score": interaction.get("score", 0),
                    "nscore": interaction.get("nscore", 0),  # Neighborhood
                    "fscore": interaction.get("fscore", 0),  # Fusion
                    "pscore": interaction.get("pscore", 0),  # Phylogenetic
                    "ascore": interaction.get("ascore", 0),  # Co-expression (text)
                    "escore": interaction.get("escore", 0),  # Experimental
                    "dscore": interaction.get("dscore", 0),  # Database
                    "tscore": interaction.get("tscore", 0),  # Textmining
                })
            
            # Sort by score
            partners.sort(key=lambda x: x["combined_score"], reverse=True)
            
            return {
                "query_protein": protein,
                "species": species_id,
                "n_partners": len(partners),
                "partners": partners,
            }
        return result
    
    def get_functional_enrichment(
        self,
        proteins: List[str],
        species: str = "human",
    ) -> Dict[str, Any]:
        """
        Get functional enrichment analysis for a set of proteins.
        
        Returns enriched GO terms, KEGG pathways, Pfam domains, etc.
        
        Args:
            proteins: List of protein/gene names
            species: Species name
            
        Returns:
            Dict with enriched terms grouped by category
        """
        if isinstance(species, str) and species.lower() in self.SPECIES:
            species_id = self.SPECIES[species.lower()]
        else:
            species_id = 9606
        
        params = {
            "identifiers": "%0d".join(proteins),
            "species": species_id,
        }
        
        result = self._request("enrichment", params)
        
        if isinstance(result, list):
            # Group by category
            grouped = {}
            for term in result:
                category = term.get("category", "Unknown")
                if category not in grouped:
                    grouped[category] = []
                grouped[category].append({
                    "term": term.get("term", ""),
                    "description": term.get("description", ""),
                    "p_value": term.get("p_value", 1.0),
                    "fdr": term.get("fdr", 1.0),
                    "genes_in_term": term.get("number_of_genes", 0),
                    "genes_matched": term.get("number_of_genes_in_background", 0),
                    "input_genes": term.get("inputGenes", ""),
                })
            
            # Sort each category by p-value
            for cat in grouped:
                grouped[cat].sort(key=lambda x: x["p_value"])
            
            return {
                "query_proteins": proteins,
                "n_proteins": len(proteins),
                "enrichment_by_category": grouped,
            }
        return result
    
    def get_protein_info(
        self,
        proteins: List[str],
        species: str = "human",
    ) -> Dict[str, Any]:
        """
        Get detailed information about proteins.
        
        Args:
            proteins: List of protein/gene names
            species: Species name
            
        Returns:
            Dict with protein annotations
        """
        if isinstance(species, str) and species.lower() in self.SPECIES:
            species_id = self.SPECIES[species.lower()]
        else:
            species_id = 9606
        
        params = {
            "identifiers": "%0d".join(proteins),
            "species": species_id,
        }
        
        # First resolve the identifiers
        result = self._request("resolve", params)
        
        if isinstance(result, list):
            return {
                "query": proteins,
                "resolved": [
                    {
                        "query": r.get("queryItem", ""),
                        "string_id": r.get("stringId", ""),
                        "preferred_name": r.get("preferredName", ""),
                        "taxon": r.get("taxonName", ""),
                        "annotation": r.get("annotation", ""),
                    }
                    for r in result
                ]
            }
        return result
    
    def get_network_image_url(
        self,
        proteins: List[str],
        species: str = "human",
        required_score: int = 400,
        network_type: str = "functional",
    ) -> str:
        """
        Get URL for network visualization image.
        
        Args:
            proteins: List of proteins
            species: Species name
            required_score: Minimum score
            network_type: Network type
            
        Returns:
            URL to PNG image of the network
        """
        if isinstance(species, str) and species.lower() in self.SPECIES:
            species_id = self.SPECIES[species.lower()]
        else:
            species_id = 9606
        
        params = {
            "identifiers": "%0d".join(proteins),
            "species": species_id,
            "required_score": required_score,
            "network_type": network_type,
        }
        
        # Build URL (image endpoint)
        param_str = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{self.BASE_URL}/image/network?{param_str}"


# ============================================================================
# KEGG DATABASE (Pathways)
# ============================================================================

class KEGGClient:
    """
    Client for KEGG (Kyoto Encyclopedia of Genes and Genomes) API.
    
    https://www.kegg.jp/
    
    API Documentation: https://www.kegg.jp/kegg/rest/keggapi.html
    """
    
    BASE_URL = "https://rest.kegg.jp"
    
    # Common organism codes
    ORGANISMS = {
        "human": "hsa",
        "mouse": "mmu",
        "rat": "rno",
        "zebrafish": "dre",
        "fly": "dme",
        "worm": "cel",
        "yeast": "sce",
        "ecoli": "eco",
    }
    
    def _request(self, operation: str, *args) -> str:
        """Make KEGG API request."""
        path = "/".join([operation] + list(args))
        url = f"{self.BASE_URL}/{path}"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"KEGG API error: {e}")
            return f"ERROR: {e}"
    
    def _parse_flat(self, text: str) -> List[Dict[str, str]]:
        """Parse tab-separated KEGG response."""
        results = []
        for line in text.strip().split("\n"):
            if "\t" in line:
                parts = line.split("\t")
                if len(parts) >= 2:
                    results.append({"id": parts[0], "description": parts[1]})
        return results
    
    def _parse_entry(self, text: str) -> Dict[str, Any]:
        """Parse KEGG flat file entry format."""
        entry = {}
        current_key = None
        current_value = []
        
        for line in text.split("\n"):
            if not line.strip():
                continue
            
            # Check if this is a new key (starts at column 0, not whitespace)
            if line[0] != " " and line[0] != "\t":
                # Save previous key/value
                if current_key:
                    entry[current_key] = "\n".join(current_value).strip()
                
                # Parse new key
                if " " in line:
                    parts = line.split(None, 1)
                    current_key = parts[0]
                    current_value = [parts[1]] if len(parts) > 1 else []
                else:
                    current_key = line.strip()
                    current_value = []
            else:
                # Continuation of previous value
                current_value.append(line.strip())
        
        # Don't forget the last key
        if current_key:
            entry[current_key] = "\n".join(current_value).strip()
        
        return entry
    
    def search_pathways(
        self,
        query: str,
        organism: str = "human",
    ) -> Dict[str, Any]:
        """
        Search for pathways by keyword.
        
        Args:
            query: Search term (e.g., "apoptosis", "cell cycle", "cancer")
            organism: Organism name or KEGG code
            
        Returns:
            Dict with matching pathways
        """
        # Resolve organism
        if organism.lower() in self.ORGANISMS:
            org_code = self.ORGANISMS[organism.lower()]
        else:
            org_code = organism.lower()
        
        # Search in pathway database
        result = self._request("find", "pathway", query)
        
        if result.startswith("ERROR"):
            return {"error": result}
        
        pathways = self._parse_flat(result)
        
        # Filter to organism if specified
        organism_pathways = []
        for p in pathways:
            # KEGG pathway IDs like "path:hsa04110"
            pid = p["id"].replace("path:", "")
            if pid.startswith(org_code) or pid.startswith("map"):
                organism_pathways.append({
                    "pathway_id": pid,
                    "name": p["description"],
                    "url": f"https://www.kegg.jp/kegg-bin/show_pathway?{pid}",
                })
        
        return {
            "query": query,
            "organism": org_code,
            "n_pathways": len(organism_pathways),
            "pathways": organism_pathways,
        }
    
    def get_pathway(
        self,
        pathway_id: str,
    ) -> Dict[str, Any]:
        """
        Get detailed information about a pathway.
        
        Args:
            pathway_id: KEGG pathway ID (e.g., "hsa04110" for cell cycle)
            
        Returns:
            Dict with pathway details including genes
        """
        # Clean up pathway ID
        pathway_id = pathway_id.replace("path:", "").strip()
        
        result = self._request("get", f"path:{pathway_id}")
        
        if result.startswith("ERROR"):
            return {"error": result}
        
        entry = self._parse_entry(result)
        
        # Parse genes section
        genes = []
        if "GENE" in entry:
            for line in entry["GENE"].split("\n"):
                parts = line.split(None, 1)
                if len(parts) >= 2:
                    gene_id = parts[0]
                    # Gene description often has format "SYMBOL; description [KO:xxx]"
                    desc = parts[1]
                    if ";" in desc:
                        symbol, rest = desc.split(";", 1)
                    else:
                        symbol = desc.split()[0] if desc else gene_id
                        rest = desc
                    genes.append({
                        "gene_id": gene_id,
                        "symbol": symbol.strip(),
                        "description": rest.strip(),
                    })
        
        return {
            "pathway_id": pathway_id,
            "name": entry.get("NAME", "").replace(" - Homo sapiens (human)", "").strip(),
            "description": entry.get("DESCRIPTION", ""),
            "class": entry.get("CLASS", ""),
            "n_genes": len(genes),
            "genes": genes[:100],  # Limit for display
            "url": f"https://www.kegg.jp/kegg-bin/show_pathway?{pathway_id}",
        }
    
    def get_genes_in_pathway(
        self,
        pathway_id: str,
    ) -> Dict[str, Any]:
        """
        Get all genes in a pathway.
        
        Args:
            pathway_id: KEGG pathway ID
            
        Returns:
            Dict with gene list
        """
        pathway_id = pathway_id.replace("path:", "").strip()
        
        result = self._request("link", "genes", f"path:{pathway_id}")
        
        if result.startswith("ERROR"):
            return {"error": result}
        
        genes = []
        for line in result.strip().split("\n"):
            if "\t" in line:
                parts = line.split("\t")
                if len(parts) >= 2:
                    gene_id = parts[1].replace("hsa:", "").replace("mmu:", "")
                    genes.append(gene_id)
        
        return {
            "pathway_id": pathway_id,
            "n_genes": len(genes),
            "gene_ids": genes,
        }
    
    def find_pathways_for_gene(
        self,
        gene: str,
        organism: str = "human",
    ) -> Dict[str, Any]:
        """
        Find all pathways containing a gene.
        
        Args:
            gene: Gene symbol (e.g., "TP53") or KEGG gene ID
            organism: Organism name
            
        Returns:
            Dict with pathways containing the gene
        """
        if organism.lower() in self.ORGANISMS:
            org_code = self.ORGANISMS[organism.lower()]
        else:
            org_code = organism.lower()
        
        # First, try to find the gene
        gene_result = self._request("find", "genes", f"{gene}")
        
        if gene_result.startswith("ERROR"):
            return {"error": gene_result}
        
        # Find the right gene ID for the organism
        kegg_gene_id = None
        for line in gene_result.strip().split("\n"):
            if "\t" in line:
                gid = line.split("\t")[0]
                if gid.startswith(f"{org_code}:"):
                    kegg_gene_id = gid
                    break
        
        if not kegg_gene_id:
            # Try direct format
            kegg_gene_id = f"{org_code}:{gene}"
        
        # Get pathways for this gene
        pathway_result = self._request("link", "pathway", kegg_gene_id)
        
        if pathway_result.startswith("ERROR") or not pathway_result.strip():
            return {
                "gene": gene,
                "kegg_id": kegg_gene_id,
                "n_pathways": 0,
                "pathways": [],
            }
        
        # Parse pathways
        pathway_ids = []
        for line in pathway_result.strip().split("\n"):
            if "\t" in line:
                parts = line.split("\t")
                if len(parts) >= 2:
                    pid = parts[1].replace("path:", "")
                    if pid.startswith(org_code):
                        pathway_ids.append(pid)
        
        # Get pathway names
        pathways = []
        for pid in pathway_ids[:20]:  # Limit to avoid too many requests
            info_result = self._request("list", f"path:{pid}")
            if not info_result.startswith("ERROR"):
                for line in info_result.strip().split("\n"):
                    if "\t" in line:
                        name = line.split("\t")[1]
                        pathways.append({
                            "pathway_id": pid,
                            "name": name.replace(" - Homo sapiens (human)", "").strip(),
                            "url": f"https://www.kegg.jp/kegg-bin/show_pathway?{pid}+{gene}",
                        })
                        break
        
        return {
            "gene": gene,
            "kegg_id": kegg_gene_id,
            "organism": org_code,
            "n_pathways": len(pathways),
            "pathways": pathways,
        }
    
    def find_pathways_for_genes(
        self,
        genes: List[str],
        organism: str = "human",
    ) -> Dict[str, Any]:
        """
        Find pathways containing multiple genes (pathway enrichment-like).
        
        Args:
            genes: List of gene symbols
            organism: Organism name
            
        Returns:
            Dict with pathways and gene overlap counts
        """
        if organism.lower() in self.ORGANISMS:
            org_code = self.ORGANISMS[organism.lower()]
        else:
            org_code = organism.lower()
        
        # Collect pathways for each gene
        pathway_counts = {}  # pathway_id -> {name, genes}
        
        for gene in genes[:20]:  # Limit to avoid rate limiting
            result = self.find_pathways_for_gene(gene, organism)
            if "pathways" in result:
                for p in result["pathways"]:
                    pid = p["pathway_id"]
                    if pid not in pathway_counts:
                        pathway_counts[pid] = {
                            "name": p["name"],
                            "genes": [],
                            "url": p["url"],
                        }
                    pathway_counts[pid]["genes"].append(gene)
        
        # Sort by number of genes
        sorted_pathways = sorted(
            pathway_counts.items(),
            key=lambda x: len(x[1]["genes"]),
            reverse=True
        )
        
        pathways = [
            {
                "pathway_id": pid,
                "name": data["name"],
                "genes_in_input": data["genes"],
                "n_genes_matched": len(data["genes"]),
                "url": data["url"],
            }
            for pid, data in sorted_pathways
        ]
        
        return {
            "query_genes": genes,
            "n_genes_queried": len(genes),
            "n_pathways_found": len(pathways),
            "pathways": pathways,
        }
    
    def get_disease_pathways(
        self,
        disease: str,
    ) -> Dict[str, Any]:
        """
        Search for disease-related pathways.
        
        Args:
            disease: Disease name (e.g., "cancer", "diabetes", "alzheimer")
            
        Returns:
            Dict with disease pathways
        """
        result = self._request("find", "pathway", disease)
        
        if result.startswith("ERROR"):
            return {"error": result}
        
        pathways = self._parse_flat(result)
        
        formatted = [
            {
                "pathway_id": p["id"].replace("path:", ""),
                "name": p["description"],
                "url": f"https://www.kegg.jp/kegg-bin/show_pathway?{p['id'].replace('path:', '')}",
            }
            for p in pathways
        ]
        
        return {
            "query": disease,
            "n_pathways": len(formatted),
            "pathways": formatted,
        }


# ============================================================================
# FORMAT FUNCTIONS FOR AGENT OUTPUT
# ============================================================================

def format_string_interactions(result: Dict[str, Any]) -> str:
    """Format STRING interaction results for agent."""
    if "error" in result:
        return f"❌ Error: {result['error']}"
    
    lines = []
    lines.append("=" * 60)
    lines.append("**STRING Protein-Protein Interactions**")
    lines.append("=" * 60)
    
    if "query_protein" in result:
        lines.append(f"Query: {result['query_protein']}")
        lines.append(f"Partners found: {result.get('n_partners', 0)}")
        lines.append("")
        lines.append("**Interaction Partners:**")
        lines.append("-" * 40)
        
        for p in result.get("partners", [])[:15]:
            score = p.get("combined_score", 0)
            # Score interpretation
            if score >= 900:
                confidence = "highest"
            elif score >= 700:
                confidence = "high"
            elif score >= 400:
                confidence = "medium"
            else:
                confidence = "low"
            
            evidence = []
            if p.get("escore", 0) > 0: evidence.append("experimental")
            if p.get("dscore", 0) > 0: evidence.append("database")
            if p.get("tscore", 0) > 0: evidence.append("textmining")
            
            lines.append(f"• {p['partner']}: score={score} ({confidence} confidence)")
            if evidence:
                lines.append(f"    Evidence: {', '.join(evidence)}")
    
    elif "interactions" in result:
        lines.append(f"Query proteins: {', '.join(result.get('query_proteins', []))}")
        lines.append(f"Interactions found: {result.get('n_interactions', 0)}")
        lines.append(f"Min score threshold: {result.get('min_score', 400)}")
        lines.append("")
        lines.append("**Interactions:**")
        lines.append("-" * 40)
        
        for i in result.get("interactions", [])[:20]:
            prot_a = i.get("preferredName_A", i.get("stringId_A", "?"))
            prot_b = i.get("preferredName_B", i.get("stringId_B", "?"))
            score = i.get("score", 0)
            lines.append(f"• {prot_a} ↔ {prot_b}: score={score}")
    
    lines.append("")
    lines.append("**Score interpretation:**")
    lines.append("• ≥900: Highest confidence")
    lines.append("• ≥700: High confidence")
    lines.append("• ≥400: Medium confidence")
    lines.append("• <400: Low confidence")
    
    return "\n".join(lines)


def format_string_enrichment(result: Dict[str, Any]) -> str:
    """Format STRING enrichment results for agent."""
    if "error" in result:
        return f"❌ Error: {result['error']}"
    
    lines = []
    lines.append("=" * 60)
    lines.append("**STRING Functional Enrichment Analysis**")
    lines.append("=" * 60)
    lines.append(f"Query proteins: {', '.join(result.get('query_proteins', []))}")
    lines.append("")
    
    for category, terms in result.get("enrichment_by_category", {}).items():
        lines.append(f"**{category}:**")
        lines.append("-" * 40)
        
        for term in terms[:5]:  # Top 5 per category
            fdr = term.get("fdr", 1)
            if fdr < 0.001:
                sig = "***"
            elif fdr < 0.01:
                sig = "**"
            elif fdr < 0.05:
                sig = "*"
            else:
                sig = ""
            
            lines.append(f"• {term.get('description', term.get('term', '?'))}{sig}")
            lines.append(f"    FDR: {fdr:.2e}, Genes: {term.get('input_genes', '')}")
        
        lines.append("")
    
    lines.append("Significance: *** p<0.001, ** p<0.01, * p<0.05")
    
    return "\n".join(lines)


def format_kegg_pathways(result: Dict[str, Any]) -> str:
    """Format KEGG pathway results for agent."""
    if "error" in result:
        return f"❌ Error: {result['error']}"
    
    lines = []
    lines.append("=" * 60)
    lines.append("**KEGG Pathway Search Results**")
    lines.append("=" * 60)
    
    if "query" in result:
        lines.append(f"Query: {result.get('query', '')}")
    if "gene" in result:
        lines.append(f"Gene: {result.get('gene', '')}")
    if "query_genes" in result:
        lines.append(f"Genes: {', '.join(result.get('query_genes', []))}")
    
    lines.append(f"Pathways found: {result.get('n_pathways', 0)}")
    lines.append("")
    lines.append("**Pathways:**")
    lines.append("-" * 40)
    
    for p in result.get("pathways", [])[:15]:
        lines.append(f"• [{p.get('pathway_id', '')}] {p.get('name', 'Unknown')}")
        if "n_genes_matched" in p:
            lines.append(f"    Genes matched: {p['n_genes_matched']} ({', '.join(p.get('genes_in_input', []))})")
        if "url" in p:
            lines.append(f"    URL: {p['url']}")
    
    return "\n".join(lines)


def format_kegg_pathway_detail(result: Dict[str, Any]) -> str:
    """Format KEGG pathway detail for agent."""
    if "error" in result:
        return f"❌ Error: {result['error']}"
    
    lines = []
    lines.append("=" * 60)
    lines.append(f"**KEGG Pathway: {result.get('name', 'Unknown')}**")
    lines.append("=" * 60)
    lines.append(f"ID: {result.get('pathway_id', '')}")
    lines.append(f"Class: {result.get('class', 'N/A')}")
    lines.append(f"Genes in pathway: {result.get('n_genes', 0)}")
    lines.append(f"URL: {result.get('url', '')}")
    lines.append("")
    
    if result.get("description"):
        lines.append("**Description:**")
        lines.append(result["description"][:500])
        lines.append("")
    
    lines.append("**Genes (first 30):**")
    lines.append("-" * 40)
    
    for g in result.get("genes", [])[:30]:
        lines.append(f"• {g.get('symbol', g.get('gene_id', '?'))}: {g.get('description', '')[:60]}")
    
    if result.get("n_genes", 0) > 30:
        lines.append(f"... and {result['n_genes'] - 30} more genes")
    
    return "\n".join(lines)


# ============================================================================
# LANGCHAIN TOOLS
# ============================================================================

try:
    from langchain_core.tools import tool
    
    # Global clients
    _string_client = None
    _kegg_client = None
    
    def _get_string() -> STRINGClient:
        global _string_client
        if _string_client is None:
            _string_client = STRINGClient()
        return _string_client
    
    def _get_kegg() -> KEGGClient:
        global _kegg_client
        if _kegg_client is None:
            _kegg_client = KEGGClient()
        return _kegg_client
    
    # =========== STRING TOOLS ===========
    
    @tool
    def string_get_interactions(
        proteins: str,
        species: str = "human",
        min_score: int = 400,
    ) -> str:
        """
        Get protein-protein interactions from STRING database.
        
        STRING contains known and predicted protein interactions from multiple
        evidence sources: experiments, databases, text mining, co-expression, etc.
        
        Args:
            proteins: Comma-separated protein/gene names (e.g., "TP53,MDM2,CDKN1A")
            species: Species name (human, mouse, rat, zebrafish, fly, worm, yeast)
            min_score: Minimum confidence score (0-1000):
                      - 900: highest confidence
                      - 700: high confidence  
                      - 400: medium confidence (default)
                      - 150: low confidence
        
        Returns:
            Interaction network with scores and evidence types.
            
        Example:
            string_get_interactions(proteins="TP53,MDM2,CDKN1A", min_score=700)
        """
        client = _get_string()
        protein_list = [p.strip() for p in proteins.split(",") if p.strip()]
        
        if len(protein_list) == 1:
            result = client.get_interaction_partners(
                protein_list[0], species, min_score
            )
        else:
            result = client.get_interactions(
                protein_list, species, min_score
            )
        
        return format_string_interactions(result)
    
    @tool
    def string_functional_enrichment(
        proteins: str,
        species: str = "human",
    ) -> str:
        """
        Perform functional enrichment analysis on a set of proteins using STRING.
        
        Identifies enriched GO terms, KEGG pathways, Pfam domains, and other
        functional categories in your protein set.
        
        Args:
            proteins: Comma-separated protein/gene names (e.g., "TP53,BRCA1,ATM,CHEK2")
            species: Species name
            
        Returns:
            Enriched functional terms grouped by category with p-values.
            
        Example:
            string_functional_enrichment(proteins="TP53,BRCA1,ATM,CHEK2,BRCA2")
        """
        client = _get_string()
        protein_list = [p.strip() for p in proteins.split(",") if p.strip()]
        
        result = client.get_functional_enrichment(protein_list, species)
        return format_string_enrichment(result)
    
    @tool
    def string_network_image(
        proteins: str,
        species: str = "human",
        min_score: int = 400,
    ) -> str:
        """
        Get URL for STRING network visualization image.
        
        Args:
            proteins: Comma-separated protein names
            species: Species name
            min_score: Minimum confidence score
            
        Returns:
            URL to PNG image of the interaction network.
        """
        client = _get_string()
        protein_list = [p.strip() for p in proteins.split(",") if p.strip()]
        
        url = client.get_network_image_url(protein_list, species, min_score)
        return f"STRING Network Image URL:\n{url}\n\nOpen this URL in a browser to view the network."
    
    # =========== KEGG TOOLS ===========
    
    @tool
    def kegg_search_pathways(
        query: str,
        organism: str = "human",
    ) -> str:
        """
        Search KEGG for pathways by keyword.
        
        KEGG (Kyoto Encyclopedia of Genes and Genomes) contains curated pathway
        maps for metabolism, signaling, disease, and more.
        
        Args:
            query: Search term (e.g., "apoptosis", "cell cycle", "MAPK", "cancer")
            organism: Organism (human, mouse, rat, etc.)
            
        Returns:
            List of matching pathways with IDs and links.
            
        Example:
            kegg_search_pathways(query="apoptosis")
            kegg_search_pathways(query="breast cancer")
        """
        client = _get_kegg()
        result = client.search_pathways(query, organism)
        return format_kegg_pathways(result)
    
    @tool
    def kegg_get_pathway(
        pathway_id: str,
    ) -> str:
        """
        Get detailed information about a KEGG pathway.
        
        Args:
            pathway_id: KEGG pathway ID (e.g., "hsa04110" for cell cycle,
                       "hsa04210" for apoptosis, "hsa05200" for pathways in cancer)
                       
        Returns:
            Pathway details including genes and description.
        """
        client = _get_kegg()
        result = client.get_pathway(pathway_id)
        return format_kegg_pathway_detail(result)
    
    @tool
    def kegg_find_pathways_for_gene(
        gene: str,
        organism: str = "human",
    ) -> str:
        """
        Find all KEGG pathways that contain a specific gene.
        
        Args:
            gene: Gene symbol (e.g., "TP53", "BRCA1", "EGFR")
            organism: Organism name
            
        Returns:
            List of pathways containing the gene with links.
            
        Example:
            kegg_find_pathways_for_gene(gene="TP53")
        """
        client = _get_kegg()
        result = client.find_pathways_for_gene(gene, organism)
        return format_kegg_pathways(result)
    
    @tool
    def kegg_find_pathways_for_genes(
        genes: str,
        organism: str = "human",
    ) -> str:
        """
        Find KEGG pathways enriched for a set of genes.
        
        Useful for understanding what biological processes a gene list is involved in.
        Returns pathways sorted by number of input genes they contain.
        
        Args:
            genes: Comma-separated gene symbols (e.g., "TP53,BRCA1,ATM,CHEK2")
            organism: Organism name
            
        Returns:
            Pathways ranked by gene overlap.
            
        Example:
            kegg_find_pathways_for_genes(genes="TP53,MDM2,CDKN1A,BAX,BCL2")
        """
        client = _get_kegg()
        gene_list = [g.strip() for g in genes.split(",") if g.strip()]
        result = client.find_pathways_for_genes(gene_list, organism)
        return format_kegg_pathways(result)
    
    @tool
    def kegg_disease_pathways(
        disease: str,
    ) -> str:
        """
        Search for disease-related pathways in KEGG.
        
        Args:
            disease: Disease name (e.g., "cancer", "diabetes", "alzheimer", 
                    "parkinson", "leukemia", "breast cancer")
                    
        Returns:
            Disease-related pathways.
            
        Example:
            kegg_disease_pathways(disease="lung cancer")
        """
        client = _get_kegg()
        result = client.get_disease_pathways(disease)
        return format_kegg_pathways(result)

except ImportError:
    pass


# ============================================================================
# STANDALONE USAGE
# ============================================================================

if __name__ == "__main__":
    # Test STRING
    print("Testing STRING...")
    string_client = STRINGClient()
    
    result = string_client.get_interaction_partners("TP53", "human", 700)
    print(format_string_interactions(result))
    
    print("\n" + "="*60 + "\n")
    
    # Test KEGG
    print("Testing KEGG...")
    kegg_client = KEGGClient()
    
    result = kegg_client.find_pathways_for_gene("TP53", "human")
    print(format_kegg_pathways(result))
