import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
from Bio import Entrez
from datetime import date
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_ollama import ChatOllama 
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
import re
import xml.etree.ElementTree as ET
from langchain_core.tools import tool
import json
from typing import Tuple, List, Optional, Dict, Any
import numpy as np
from langchain_core.prompts import PromptTemplate
import torch
import asyncio
import aiohttp
from collections import defaultdict
from pathlib import Path
import logging
import requests
from pathway_tools import (
    # STRING
    string_get_interactions,
    string_functional_enrichment,
    string_network_image,
    # KEGG
    kegg_search_pathways,
    kegg_get_pathway,
    kegg_find_pathways_for_gene,
    kegg_find_pathways_for_genes,
    kegg_disease_pathways,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Constants
REQUEST_TIMEOUT = 15

load_dotenv()
Entrez.email = "rafailadam46@gmail.com"

# ============================================
# CONFIGURATION
# ============================================

# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
logger.info(f"Script directory: {SCRIPT_DIR}")

# ============================================
# LLM
# ============================================

pi_llm = ChatOllama(model="ministral-3:8b")

# =========================================
# GENE INFO TOOL
# =========================================

@tool
def gene_info_tool(gene_symbol: str) -> str:
    """
    Get authoritative information about a gene from NCBI and UniProt.
    
    ALWAYS use this tool FIRST before answering questions about a gene's
    function, expression, or role in disease. This prevents hallucination.
    
    Args:
        gene_symbol: Gene symbol (e.g., "EGFR", "TP53", "EGFLAM")
    
    Returns:
        Official gene name, aliases, function summary, and associated diseases
    """
    # Normalize gene symbol
    gene_symbol = gene_symbol.strip().upper()
    
    output = [f"**Gene Information: {gene_symbol}**\n"]
    sources_status = {"NCBI Gene": "❌ NOT QUERIED", "UniProt": "❌ NOT QUERIED"}
    found_info = False
    
    # =========================================
    # 1. NCBI Gene Database
    # =========================================
    try:
        search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        search_params = {
            "db": "gene",
            "term": f"{gene_symbol}[Gene Name] AND Homo sapiens[Organism]",
            "retmode": "json",
            "retmax": 1
        }
        
        r = requests.get(search_url, params=search_params, timeout=REQUEST_TIMEOUT)
        
        if r.ok:
            search_data = r.json()
            id_list = search_data.get("esearchresult", {}).get("idlist", [])
            
            if id_list:
                gene_id = id_list[0]
                
                fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
                fetch_params = {"db": "gene", "id": gene_id, "retmode": "json"}
                
                r2 = requests.get(fetch_url, params=fetch_params, timeout=REQUEST_TIMEOUT)
                
                if r2.ok:
                    data = r2.json()
                    gene_data = data.get("result", {}).get(gene_id, {})
                    
                    if gene_data:
                        found_info = True
                        sources_status["NCBI Gene"] = "✅ SUCCESS"
                        
                        output.append(f"**Official Symbol:** {gene_data.get('name', gene_symbol)}")
                        output.append(f"**Full Name:** {gene_data.get('description', 'Unknown')}")
                        
                        aliases = gene_data.get("otheraliases", "")
                        if aliases:
                            output.append(f"**Aliases:** {aliases}")
                        
                        summary = gene_data.get("summary", "")
                        if summary:
                            if len(summary) > 500:
                                summary = summary[:500] + "..."
                            output.append(f"\n**Function:**\n{summary}")
                        
                        chrom = gene_data.get("chromosome", "")
                        if chrom:
                            output.append(f"\n**Location:** Chromosome {chrom}")
                        
                        output.append(f"**NCBI Gene ID:** {gene_id}")
                    else:
                        sources_status["NCBI Gene"] = "⚠️ NO DATA (empty response)"
                else:
                    sources_status["NCBI Gene"] = f"❌ FAILED (HTTP {r2.status_code})"
            else:
                sources_status["NCBI Gene"] = "⚠️ GENE NOT FOUND"
        else:
            sources_status["NCBI Gene"] = f"❌ FAILED (HTTP {r.status_code})"
            
    except requests.exceptions.Timeout:
        sources_status["NCBI Gene"] = "❌ TIMEOUT"
    except Exception as e:
        sources_status["NCBI Gene"] = f"❌ ERROR: {str(e)[:50]}"
    
    # =========================================
    # 2. UniProt
    # =========================================
    try:
        uniprot_url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            "query": f"gene_exact:{gene_symbol} AND organism_id:9606 AND reviewed:true",
            "fields": "accession,protein_name,cc_function,cc_disease",
            "format": "json",
            "size": 1
        }
        headers = {"Accept": "application/json"}
        
        r = requests.get(uniprot_url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        
        if r.ok:
            data = r.json()
            results = data.get("results", [])
            
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
                
                comments = entry.get("comments", [])
                for comment in comments:
                    if comment.get("commentType") == "FUNCTION":
                        texts = comment.get("texts", [])
                        if texts:
                            func_text = texts[0].get("value", "")
                            if func_text and "Function:" not in "\n".join(output):
                                if len(func_text) > 400:
                                    func_text = func_text[:400] + "..."
                                output.append(f"\n**Protein Function:**\n{func_text}")
                    
                    if comment.get("commentType") == "DISEASE":
                        disease = comment.get("disease", {})
                        disease_name = disease.get("diseaseId", "")
                        if disease_name:
                            output.append(f"\n**Associated Disease:** {disease_name}")
                            break
            else:
                sources_status["UniProt"] = "⚠️ GENE NOT FOUND"
        else:
            sources_status["UniProt"] = f"❌ FAILED (HTTP {r.status_code})"
            
    except requests.exceptions.Timeout:
        sources_status["UniProt"] = "❌ TIMEOUT"
    except Exception as e:
        sources_status["UniProt"] = f"❌ ERROR: {str(e)[:50]}"
    
    # =========================================
    # DATA SOURCE STATUS SUMMARY
    # =========================================
    output.append("\n" + "="*50)
    output.append("**DATA SOURCE STATUS:**")
    for source, status in sources_status.items():
        output.append(f"  • {source}: {status}")
    
    if not found_info:
        output.append("\n❌ **NO AUTHORITATIVE DATA RETRIEVED**")
        output.append(f"Gene '{gene_symbol}' was not found in any database.")
        output.append("⚠️ DO NOT make claims about this gene - admit you cannot find information.")
    
    return "\n".join(output)


# =========================================
# GENE TISSUE EXPRESSION TOOL
# =========================================

@tool
def gene_tissue_expression_tool(gene_symbol: str, tissue: str = None) -> str:
    """
    Query gene expression from GTEx (bulk RNA-seq across human tissues).
    
    This tool provides tissue-level expression data from the Genotype-Tissue 
    Expression (GTEx) project. It measures average expression across all cells
    in a tissue sample (bulk RNA-seq), which is complementary to single-cell data.
    
    Use this when:
    - You need tissue-level expression quantification (TPM values)
    - Comparing expression across different tissues
    - Checking if a gene is broadly expressed in a tissue
    - GTEx data is the gold standard for tissue transcriptomics
    
    Args:
        gene_symbol: Gene name (e.g., "TERT", "EGFR", "TP53") or Ensembl ID
        tissue: Optional tissue filter (e.g., "Lung", "Brain", "Liver", "Heart")
    
    Returns:
        Tissue expression levels in TPM (Transcripts Per Million):
        - Median expression across tissue samples
        - Rank of tissues by expression level
        - Sample size (number of donors)
    
    Example queries:
        - "TERT expression across all tissues"
        - "EGFR levels in brain vs lung"
        - "Which tissues express TP53 highest?"
    
    Note: GTEx measures bulk tissue averages. 
    """
    input_symbol = gene_symbol.strip()
    output = [f"**GTEx Tissue Expression Analysis (Bulk RNA-seq)**"]
    output.append(f"Gene: {input_symbol}")
    if tissue:
        output.append(f"Tissue focus: {tissue}")
    output.append("")
    
    # =========================================
    # STEP 1: Resolve gene symbol to GTEx Gencode ID
    # =========================================
    gtex_id = None
    official_symbol = input_symbol.upper()
    
    try:
        gtex_gene_url = "https://gtexportal.org/api/v2/reference/gene"
        params = {
            "geneId": input_symbol.upper(),
            "gencodeVersion": "v39",  # GTEx v10 uses Gencode v39
            "genomeBuild": "GRCh38/hg38",
            "page": 0,
            "itemsPerPage": 10  # Allow multiple matches for ambiguous symbols
        }
        headers = {"Accept": "application/json"}
        
        r_gtex = requests.get(gtex_gene_url, params=params, headers=headers, timeout=10)
        
        if r_gtex.ok:
            data = r_gtex.json()
            gene_data = data.get("data", [])
            
            if gene_data:
                # Take the first (best) match
                best_match = gene_data[0]
                gtex_id = best_match.get("gencodeId")  # Versioned ID: ENSG...XX.X
                official_symbol = best_match.get("geneSymbol", input_symbol.upper())
                
                # Get additional info if available
                description = best_match.get("description", "")
                
                output.append(f"✅ Resolved: {official_symbol}")
                output.append(f"   GTEx ID: {gtex_id}")
                if description:
                    output.append(f"   Description: {description}")
                output.append("")
            else:
                output.append(f"⚠️  Warning: '{input_symbol}' not found in GTEx reference database.")
                output.append(f"   The gene may not be protein-coding or may have an alternative symbol.")
                output.append(f"   Try using the official HGNC symbol or Ensembl ID.")
                output.append("")
        else:
            output.append(f"⚠️  Warning: GTEx gene lookup failed (HTTP {r_gtex.status_code}).")
            output.append(f"   Cannot resolve symbol to GTEx ID.")
            output.append("")
            
    except requests.exceptions.Timeout:
        output.append(f"❌ Error: GTEx gene lookup timed out.")
        return "\n".join(output)
    except Exception as e:
        output.append(f"❌ Error resolving gene: {str(e)[:100]}")
        return "\n".join(output)

    # =========================================
    # STEP 2: Query GTEx expression data
    # =========================================
    if not gtex_id:
        output.append("❌ Cannot query expression without valid GTEx ID.")
        output.append("\nSuggestions:")
        output.append("- Check the gene symbol spelling")
        output.append("- Try using the Ensembl ID (e.g., ENSG00000164318.9)")
        output.append("- Use gene_info_tool to find the official gene symbol")
        return "\n".join(output)
    
    try:
        exp_url = "https://gtexportal.org/api/v2/expression/medianGeneExpression"
        exp_params = {
            "gencodeId": gtex_id,  # Versioned Gencode ID required
            "datasetId": "gtex_v10",
            "format": "json"
        }
        
        headers = {"Accept": "application/json"}
        
        output.append(f"Querying GTEx v10 expression data...")
        
        r_exp = requests.get(exp_url, params=exp_params, headers=headers, timeout=15)
        
        if not r_exp.ok:
            output.append(f"❌ GTEx API error: HTTP {r_exp.status_code}")
            if r_exp.status_code == 400:
                output.append(f"   The Gencode ID may be invalid or not present in GTEx.")
            return "\n".join(output)
        
        data = r_exp.json()
        expressions = data.get("data", [])
        
        if not expressions:
            output.append(f"⚠️  No expression data returned for {official_symbol}.")
            output.append(f"   This gene may not be expressed or may be below detection threshold.")
            return "\n".join(output)
        
        # =========================================
        # STEP 3: Process and display results
        # =========================================
        output.append(f"✅ Found expression data across {len(expressions)} tissues\n")
        
        # Sort by median expression (descending)
        sorted_exp = sorted(expressions, key=lambda x: x.get("median", 0), reverse=True)
        
        # Filter by tissue if specified
        if tissue:
            tissue_lower = tissue.lower()
            filtered_exp = [
                e for e in sorted_exp 
                if tissue_lower in e.get("tissueSiteDetailId", "").lower()
            ]
            
            if filtered_exp:
                output.append(f"**Expression in '{tissue}' tissues:**")
                output.append(f"{'Tissue':<35} {'Median TPM':>12} {'n':>5}")
                output.append(f"{'-'*35} {'-'*12} {'-'*5}")
                
                for exp in filtered_exp[:10]:  # Show top 10 matches
                    tissue_name = exp.get("tissueSiteDetailId", "Unknown")[:34]
                    median = exp.get("median", 0)
                    n_samples = exp.get("nSamples", "N/A")
                    
                    output.append(
                        f"{tissue_name:<35} "
                        f"{median:>11.2f} "
                        f"{str(n_samples):>5}"
                    )
                
                if len(filtered_exp) > 10:
                    output.append(f"\n... and {len(filtered_exp) - 10} more tissue subtypes")
                    
            else:
                output.append(f"⚠️  No tissues matching '{tissue}' found.")
                output.append(f"   Available tissues include:")
                # Show top 5 tissues as suggestions
                for exp in sorted_exp[:5]:
                    output.append(f"   • {exp.get('tissueSiteDetailId')}")
        
        else:
            # Show top expressed tissues
            output.append(f"**Top 10 Tissues by Expression:**")
            output.append(f"{'Rank':<5} {'Tissue':<35} {'Median TPM':>12} {'n':>5}")
            output.append(f"{'-'*5} {'-'*35} {'-'*12} {'-'*5}")
            
            for rank, exp in enumerate(sorted_exp[:10], 1):
                tissue_name = exp.get("tissueSiteDetailId", "Unknown")[:34]
                median = exp.get("median", 0)
                n_samples = exp.get("nSamples", "N/A")
                
                output.append(
                    f"{rank:<5} "
                    f"{tissue_name:<35} "
                    f"{median:>11.2f} "
                    f"{str(n_samples):>5}"
                )
            
            # Summary statistics
            all_medians = [e.get("median", 0) for e in expressions]
            output.append(f"\n**Summary Statistics:**")
            output.append(f"  Tissues analyzed: {len(expressions)}")
            output.append(f"  Highest expression: {sorted_exp[0].get('tissueSiteDetailId')} ({sorted_exp[0].get('median'):.2f} TPM)")
            output.append(f"  Lowest expression: {sorted_exp[-1].get('tissueSiteDetailId')} ({sorted_exp[-1].get('median'):.2f} TPM)")
        
        # Add interpretation
        output.append(f"\n**Interpretation:**")
        output.append(f"• TPM = Transcripts Per Million (normalized for sequencing depth)")
        output.append(f"• Bulk RNA-seq measures average across all cells in tissue")
        
        # Highlight if expression is tissue-specific or ubiquitous
        top_tpm = sorted_exp[0].get("median", 0)
        median_tpm = sorted(all_medians)[len(all_medians)//2] if all_medians else 0
        
        if top_tpm > 10 * median_tpm and top_tpm > 10:
            output.append(f"• **Tissue-specific**: High expression in {sorted_exp[0].get('tissueSiteDetailId')} with low expression elsewhere")
        elif median_tpm > 1:
            output.append(f"• **Broadly expressed**: Detected across most tissues")
        else:
            output.append(f"• **Low expression**: Generally low expression across tissues")
        
        return "\n".join(output)
        
    except requests.exceptions.Timeout:
        output.append(f"❌ Error: GTEx expression query timed out.")
        return "\n".join(output)
    except Exception as e:
        output.append(f"❌ Error querying GTEx: {str(e)[:100]}")
        return "\n".join(output)
    
# =========================================
# GENE COORDINATES TOOL
# =========================================

@tool
def get_gene_coordinates_tool(gene_symbol: str) -> str:
    """
    Get genomic coordinates for a gene (useful for regulatory/TE queries).
    
    Args:
        gene_symbol: Gene name (e.g., "TP53", "EGFR")
    
    Returns:
        Chromosome, start, end coordinates (GRCh38) with DATA SOURCE STATUS
    """
    # Normalize gene symbol
    gene_symbol = gene_symbol.strip().upper()
    
    output = []
    sources_status = {"Ensembl REST API": "❌ NOT QUERIED"}
    
    try:
        url = f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene_symbol}"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        
        if r.ok:
            data = r.json()
            chrom = data.get("seq_region_name", "Unknown")
            start = data.get("start", 0)
            end = data.get("end", 0)
            strand = "+" if data.get("strand", 1) == 1 else "-"
            
            sources_status["Ensembl REST API"] = "✅ SUCCESS"
            
            output.append(f"**{gene_symbol}** coordinates (GRCh38):")
            output.append(f"  Chromosome: {chrom}")
            output.append(f"  Start: {start}")
            output.append(f"  End: {end}")       
            output.append(f"  Strand: {strand}")
            output.append(f"  Ensembl ID: {data.get('id', 'N/A')}")
        elif r.status_code == 404:
            sources_status["Ensembl REST API"] = "⚠️ GENE NOT FOUND"
            output.append(f"Gene '{gene_symbol}' not found in Ensembl")
        else:
            sources_status["Ensembl REST API"] = f"❌ HTTP {r.status_code}"
            output.append(f"Ensembl API error: HTTP {r.status_code}")
            
    except requests.exceptions.Timeout:
        sources_status["Ensembl REST API"] = "❌ TIMEOUT"
        output.append("Ensembl API timeout")
    except Exception as e:
        sources_status["Ensembl REST API"] = f"❌ ERROR"
        output.append(f"Error: {str(e)}")
    
    # DATA SOURCE STATUS
    output.append("\n" + "="*50)
    output.append("**DATA SOURCE STATUS:**")
    for source, status in sources_status.items():
        output.append(f"  • {source}: {status}")
    
    return "\n".join(output)

# ============================================
# AGENT SETUP
# ============================================

memory = MemorySaver()

system_prompt = """You are an advanced Biomedical Research Agent. Your goal is to provide fact-based, scientifically accurate answers using a specific set of computational tools.

# CRITICAL OPERATING RULES
1. **NO HALLUCINATION:** Never guess gene functions, expression levels, or paper citations/links. If a tool returns no data, state "No data found."
2. **VERIFY FIRST:** You must verify a gene's identity (using `gene_info_tool`) before discussing its function or expression.
3. **CITE SOURCES:** Only cite PMIDs or data sources (e.g., "GTEx v10") that explicitly appear in tool outputs.

# TOOL ROUTING GUIDE (How to choose the right tool)

## CATEGORY 1: GENE QUESTIONS ("What is GENE?", "Where is GENE expressed?")
* **Step 1: Identity (MANDATORY)**
    * Use `gene_info_tool(gene_symbol)`
    * *Goal:* Get the official symbol, summary, and aliases.
* **Step 2: Expression (If asked about tissues/cells)**
    * **Context: "Overall / Bulk tissue"** (e.g., "Is EGFR in the lung?")
        * Use `gene_tissue_expression_tool(gene_symbol, tissue)`
        * *Source:* GTEx (Bulk RNA-seq). Measures average expression in whole tissue samples.
* **Step 3: Coordinates (If asked about location)**
    * Use `get_gene_coordinates_tool(gene_symbol)`

## CATEGORY 2: PROTEIN INTERACTIONS ("What interacts with GENE?"), 
* **Protein-Protein Interactions (STRING)**
    * **Single protein:** "What interacts with TP53?"
        * Use `string_get_interactions(proteins="TP53", min_score=700)`
        * *Returns:* Interaction partners with confidence scores (0-1000)
        * *Score interpretation:* ≥900 highest, ≥700 high, ≥400 medium confidence
    * **Multiple proteins:** "Do BRCA1 and BRCA2 interact?"
        * Use `string_get_interactions(proteins="BRCA1,BRCA2,ATM", min_score=400)`
    * **Functional enrichment:** "What functions are enriched in this gene list?"
        * Use `string_functional_enrichment(proteins="TP53,MDM2,CDKN1A,BAX")`
        * *Returns:* Enriched GO terms, KEGG pathways, Pfam domains with p-values
    * **Network visualization:**
        * Use `string_network_image(proteins="TP53,MDM2,CDKN1A")` to get image URL

## CATEGORY 3: PATHWAYS ("What pathways...", "Tell me about the X pathway")
* **Pathway Analysis (KEGG)**
    * **Search by keyword:** "Find pathways related to apoptosis"
        * Use `kegg_search_pathways(query="apoptosis")`
    * **Gene → Pathways:** "What pathways is TP53 involved in?"
        * Use `kegg_find_pathways_for_gene(gene="TP53")`
    * **Gene list → Pathways:** "What pathways are these genes in: BRCA1, ATM, CHEK2?"
        * Use `kegg_find_pathways_for_genes(genes="BRCA1,ATM,CHEK2")`
        * *Returns:* Pathways ranked by number of input genes they contain
    * **Pathway details:** "Tell me about the cell cycle pathway"
        * Use `kegg_get_pathway(pathway_id="hsa04110")`
        * *Returns:* Description, gene list, and link to KEGG map
    * **Disease pathways:** "Find pathways related to breast cancer"
        * Use `kegg_disease_pathways(disease="breast cancer")`

# RESPONSE FORMATTING
* **Gene Function:** Start with the official summary from `gene_info_tool`.
* **Expression Data:**
    * Report the units (TPM for bulk).
* **Interactions (STRING):**
    * Report confidence level (high/medium/low based on score).
    * Mention evidence types if relevant (experimental, database, text-mining).
* **Pathways (KEGG):**
    * List pathway names with IDs (e.g., "Cell cycle [hsa04110]").
    * Include KEGG URLs when helpful for the user.

# EXECUTION LOOP
1. **Analyze Request:** Identify biological entities (Genes, Tissues etc).
2. **Select Tool:** Pick the tool from the Routing Guide above.
3. **Observe Output:** Read the tool's raw output.
4. **Refine/Answer:** If tool fails (e.g., "Gene not found"), try an alias. If successful, synthesize the answer.
"""

tools = [
    gene_tissue_expression_tool,
    get_gene_coordinates_tool,
    gene_info_tool,
    # STRING
    string_get_interactions,
    string_functional_enrichment,
    string_network_image,
    # KEGG
    kegg_search_pathways,
    kegg_get_pathway,
    kegg_find_pathways_for_gene,
    kegg_find_pathways_for_genes,
    kegg_disease_pathways,
]

pubmed_agent = create_agent(
    model=pi_llm,
    tools=tools,
    system_prompt=system_prompt,
    checkpointer=memory,
)

# ============================================
# MAIN
# ============================================

def main():
    print("\n" + "="*60)
    print("BioDB Agent")
    print("="*60)
    print("Commands: exit, vram, gc")
    print()
    
    while True:
        try:
            user_input = input(">>> ").strip()
            
            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break
            
            if user_input.lower() == "vram":
                alloc = torch.cuda.memory_allocated() / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"\nVRAM: {alloc:.2f} / {total:.2f} GB\n")
                continue
           
            print("\nProcessing...")
            result = pubmed_agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config={"configurable": {"thread_id": "cli"}, "recursion_limit": 500}
            )
            
            for msg in reversed(result['messages']):
                if isinstance(msg, AIMessage) and msg.content:
                    print(f"\nAI: {msg.content}\n")
                    break
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
