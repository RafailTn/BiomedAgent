"""
Unified Biomedical Agent v1.0
=============================
Single agent with the full toolset (gene, disease, expression, interaction,
pathway, genomics) — replaces research_agent.py + biodb_agent.py.

Routing is kept sharp with a domain-organized keyword table plus explicit
disambiguation rules for the known overlap pairs (bulk-vs-single-cell
expression, Open-Targets-vs-KEGG disease, STRING-vs-KEGG pathways).
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import logging
import subprocess

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
from Bio import Entrez

from bio_tools import BIO_TOOLS
from gprofiler_tool import gprofiler_enrichment_tool
from hpa_tool import hpa_protein_atlas_tool
from pathway_tools import (
    string_get_interactions,
    string_network_image,
    kegg_search_pathways,
    kegg_get_pathway,
    kegg_find_pathways_for_gene,
    kegg_disease_pathways,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
Entrez.email = "rafailadam46@gmail.com"

LLM_MODEL = "qwen3.5:9b"
# num_ctx caps prompt + generated tokens combined. Ollama's 4096 default is too
# small here: tool outputs are large Markdown blocks that accumulate in the
# MemorySaver history, so the prompt alone can reach ~4000 tokens after a couple
# of turns — leaving no room for the answer (done_reason="length") or, one turn
# later, nothing at all. 8192 doubles the room and fits entirely in VRAM on an
# 8 GB card (16384 spilled ~1.2 GB to CPU and slowed generation).
pi_llm = ChatOllama(model=LLM_MODEL, num_ctx=8192)


def unload_llm():
    """Ask Ollama to drop the model from VRAM immediately (don't wait for keep-alive)."""
    try:
        subprocess.run(["ollama", "stop", LLM_MODEL], timeout=10, capture_output=True)
        logger.info(f"Unloaded {LLM_MODEL} from Ollama.")
    except Exception as e:
        logger.debug(f"Could not unload {LLM_MODEL}: {e}")


# ===========================================================================
# SYSTEM PROMPT — domain-organized routing + explicit disambiguation
# ===========================================================================

system_prompt = """You are a biomedical research assistant. Use tools to answer; never invent data. If a tool returns nothing, say so.
If the user does not specify an organism, assume HUMAN and state that assumption in your answer.

ROUTING:
  gene function / aliases / "what is [GENE]"   → gene_info_tool  (call FIRST for any gene)
  diseases linked to a GENE                    → opentargets_associations_tool  (not KEGG, not gene_info)
  expression in a TISSUE/organ (bulk, TPM)     → gene_tissue_expression_tool
  which cell types express a gene / cell-type expression or specificity, HUMAN → hpa_protein_atlas_tool
  which cell types express a gene, NON-HUMAN (mouse, zebrafish, …) → get_gene_fraction_detected  (organ arg, never a cell type)
  marker genes that define a CELL TYPE in an organ → get_cell_type_markers
  ONLY "which cell type expresses a gene the MOST / highest / top" (ranking) → gene_highest_expression_celltype
  subcellular location / cancer prognostic marker (survival) / brain-region expression / secreted protein (HUMAN) → hpa_protein_atlas_tool
  gene coordinates / location                  → get_gene_coordinates_tool
  promoter / TSS                               → get_promoter_coordinates_tool
  ATAC / ChIP / chromatin signal prediction    → predict_genomics  (AI prediction — say so)
  protein interactions / PPI                   → string_get_interactions
  enrichment / pathways for a gene SET (2+ genes, "shared/common pathways") → gprofiler_enrichment_tool
  pathways for exactly ONE gene                → kegg_find_pathways_for_gene  (never loop it over a gene set — use gprofiler for sets)
  pathway search by keyword / details by ID    → kegg_search_pathways / kegg_get_pathway
  pathways for a DISEASE                       → kegg_disease_pathways
  network image                                → string_network_image
  availability of public datasets / "is there a GEO study/dataset on X" / "find RNA-seq data for X" → geo_search_tool  (finds studies only — NOT expression values; for expression use GTEx/HPA/AtlasApprox)
  m6A / N6-methyladenosine / RNA methylation of a GENE ("does GENE have m6A / where is GENE methylated") → m6a_modification_tool  (RNA epitranscriptomic methylation, NOT DNA methylation)

Cell-type tools: plain "which cell types express X" → hpa_protein_atlas_tool (human) or get_gene_fraction_detected (non-human); only superlative "most/highest/top" → gene_highest_expression_celltype; "markers of cell type Y" → get_cell_type_markers.

Genomic predictions: get coordinates first, then predict_genomics (chr:start-end, tissue, assays); two separate calls to compare tissues."""

tools = BIO_TOOLS + [
    hpa_protein_atlas_tool,
    gprofiler_enrichment_tool,
    string_get_interactions,
    string_network_image,
    kegg_search_pathways,
    kegg_get_pathway,
    kegg_find_pathways_for_gene,
    kegg_disease_pathways,
]

# Keep the running history from crowding out the num_ctx budget (which caused
# done_reason="length" mid-answer cutoffs). Instead of dropping old turns, this
# summarizes them: once history grows past ~4096 tokens, everything except the
# most recent ~1500 tokens is replaced by an LLM-written summary, so older
# context is compressed rather than lost. The summary is generated by the same
# local model (pi_llm) — no extra model load, but it does add one inference call
# whenever the threshold is crossed. With num_ctx=8192 this keeps the prompt
# (system + summary + recent turns) small enough to leave room for a full answer.
summarization = SummarizationMiddleware(
    model=pi_llm,
    trigger=("tokens", 4096),
    keep=("tokens", 1500),
)

memory = MemorySaver()
agent = create_agent(
    model=pi_llm,
    tools=tools,
    system_prompt=system_prompt,
    middleware=[summarization],
    checkpointer=memory,
)


def main():
    print("\n" + "=" * 60)
    print(f"Unified Biomedical Agent v1.0 - {LLM_MODEL}")
    print("=" * 60)
    print(f"{len(tools)} tools. Commands: exit, unload")
    print()

    while True:
        try:
            user_input = input(">>> ").strip()

            if user_input.lower() in {"exit", "quit"}:
                unload_llm()
                print("Goodbye!")
                break

            if user_input.lower() == "unload":
                unload_llm()
                print("LLM unloaded from VRAM\n")
                continue

            print("\nProcessing...")
            result = agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config={"configurable": {"thread_id": "cli"}, "recursion_limit": 50}
            )
            for msg in reversed(result['messages']):
                if isinstance(msg, AIMessage) and msg.content:
                    print(f"\nAI: {msg.content}\n")
                    break

        except KeyboardInterrupt:
            unload_llm()
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
