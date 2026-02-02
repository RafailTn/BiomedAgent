"""
PubMed Research Agent v3.2
==========================
Main agent script that imports KG functionality from knowledge_graph.py

Features:
- MedCPT asymmetric embeddings for biomedical retrieval
- BM25 + Dense hybrid search with RRF fusion
- Knowledge graph integration for hallucination reduction
- Async PubMed fetching with full-text support
- **Integrated Fast PrimeKG Loading**
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from datetime import date
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_ollama import ChatOllama 
from Bio import Entrez
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage
import re
import xml.etree.ElementTree as ET
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
import json
from typing import List, Optional, Dict, Any
import numpy as np
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_core.prompts import PromptTemplate
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import torch
from transformers import AutoTokenizer, AutoModel
import asyncio
import aiohttp
from collections import defaultdict
from rank_bm25 import BM25Okapi
from pathlib import Path
import logging
import requests
# Import knowledge graph module
from knowledge_graph import KnowledgeGraphManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Constants
REQUEST_TIMEOUT = 15

load_dotenv()

Entrez.email = "rafailadam46@gmail.com"


# ============================================
# CONFIGURATION
# ============================================

# Knowledge Graph settings
ENABLE_KNOWLEDGE_GRAPH = True
ENABLE_FACT_VERIFICATION = True
ENABLE_KG_ENHANCED_RETRIEVAL = True

LOAD_PRIMEKG = True
PRIMEKG_LIMIT = None  # Set to 100000 for quick testing

# GLiNER device - use "cpu" to save VRAM
GLINER_DEVICE = "cpu"

# Paths
SCRIPT_DIR = Path(__file__).parent.resolve()
KUZU_DB_PATH = SCRIPT_DIR / "kuzu_biomedical_kg"
PRIMEKG_DATA_DIR = SCRIPT_DIR / "primekg_data"
VECTORSTORE_PATH = SCRIPT_DIR / "medcpt_pubmed_rag_db"

logger.info(f"Script directory: {SCRIPT_DIR}")


# ============================================
# LLM
# ============================================

pi_llm = ChatOllama(model="ministral-3:8b")


# ============================================
# MEDCPT EMBEDDINGS
# ============================================

class MedCPTEmbeddings(Embeddings):
    """Asymmetric MedCPT embeddings for biomedical text."""
    
    def __init__(self, device: str = "cuda", batch_size: int = 8):
        self.device = device
        self.batch_size = batch_size
        
        logger.info("Loading MedCPT Query Encoder...")
        self.query_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        self.query_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(device)
        
        logger.info("Loading MedCPT Article Encoder...")
        self.article_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
        self.article_model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to(device)
        
        self.query_model.eval()
        self.article_model.eval()
        torch.cuda.empty_cache()
        logger.info("MedCPT loaded!")
    
    def _encode(self, texts: List[str], model, tokenizer) -> List[List[float]]:
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                encoded = tokenizer(batch, truncation=True, padding=True, 
                                   max_length=512, return_tensors="pt").to(self.device)
                outputs = model(**encoded)
                embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings = torch.nn.functional.normalize(embeddings, dim=1)
                all_embeddings.extend(embeddings.cpu().numpy().tolist())
                del encoded, outputs, embeddings
        return all_embeddings
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts, self.article_model, self.article_tokenizer)
    
    def embed_query(self, text: str) -> List[float]:
        return self._encode([text], self.query_model, self.query_tokenizer)[0]


# ============================================
# HYBRID RETRIEVAL
# ============================================

def reciprocal_rank_fusion(results_lists, k=60, top_n=None):
    """Fuse multiple result lists using RRF."""
    scores = defaultdict(float)
    doc_map = {}
    for results in results_lists:
        for rank, doc in enumerate(results):
            doc_id = f"{doc.metadata.get('pmid', 'x')}_{hash(doc.page_content[:100])}"
            doc_map[doc_id] = doc
            scores[doc_id] += 1.0 / (k + rank + 1)
    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[did] for did in sorted_ids][:top_n] if top_n else [doc_map[did] for did in sorted_ids]


class BM25SparseRetriever:
    """BM25 sparse retrieval."""
    
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.documents = []
        self.bm25 = None
        self.corpus = []
        self._build_index()
    
    def _build_index(self):
        try:
            collection = self.vectorstore._collection
            if collection.count() == 0:
                return
            all_docs = collection.get(include=['documents', 'metadatas'])
            self.documents = [Document(page_content=c, metadata=m or {}) 
                            for c, m in zip(all_docs.get('documents', []), all_docs.get('metadatas', [])) if c]
            self.corpus = [re.findall(r'\w+', d.page_content.lower()) for d in self.documents]
            if self.corpus:
                self.bm25 = BM25Okapi(self.corpus)
        except Exception as e:
            logger.error(f"BM25 build failed: {e}")
    
    def refresh(self):
        self._build_index()
    
    def search(self, query, k=10):
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(re.findall(r'\w+', query.lower()))
        top_idx = np.argsort(scores)[::-1][:k]
        return [self.documents[i] for i in top_idx if scores[i] > 0]


class HybridRetriever:
    """Dense + Sparse hybrid retrieval with RRF."""
    
    def __init__(self, vectorstore, bm25_retriever):
        self.vectorstore = vectorstore
        self.bm25_retriever = bm25_retriever
    
    def search(self, query, k=10):
        results = []
        try:
            dense = self.vectorstore.similarity_search(query, k=20)
            if dense:
                results.append(dense)
        except:
            pass
        try:
            sparse = self.bm25_retriever.search(query, k=20)
            if sparse:
                results.append(sparse)
        except:
            pass
        return reciprocal_rank_fusion(results, top_n=k) if results else []


# ============================================
# ASYNC PUBMED FETCHER
# ============================================

class AsyncPubMedFetcher:
    """Async PubMed API client."""
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def __init__(self, email, api_key=None, max_concurrent=3):
        self.email = email
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.semaphore = None
    
    def _params(self, **kw):
        p = {"email": self.email, **kw}
        if self.api_key:
            p["api_key"] = self.api_key
        return p
    
    async def _request(self, session, url, params):
        async with self.semaphore:
            try:
                async with session.get(url, params=params) as r:
                    if r.status == 200:
                        return await r.text()
            except:
                pass
            finally:
                await asyncio.sleep(0.35)
        return None
    
    async def search(self, term, retmax=10):
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        async with aiohttp.ClientSession() as session:
            r = await self._request(session, f"{self.BASE_URL}/esearch.fcgi", 
                                   self._params(db="pubmed", term=term, retmax=str(retmax), retmode="json"))
            if r:
                try:
                    return json.loads(r).get("esearchresult", {}).get("idlist", [])
                except:
                    pass
        return []
    
    async def fetch_abstracts(self, pmids):
        if not pmids:
            return {}
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        results = {}
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(pmids), 50):
                r = await self._request(session, f"{self.BASE_URL}/efetch.fcgi", 
                                       self._params(db="pubmed", id=",".join(pmids[i:i+50]), 
                                                   rettype="abstract", retmode="xml"))
                if r:
                    results.update(self._parse_xml(r))
        return results
    
    async def fetch_full_texts(self, pmids):
        return {p: None for p in pmids}
    
    def _parse_xml(self, xml_text):
        results = {}
        try:
            root = ET.fromstring(xml_text)
            for article in root.findall('.//PubmedArticle'):
                pmid = article.find('.//PMID')
                if pmid is None:
                    continue
                pmid = pmid.text
                title = article.find('.//ArticleTitle')
                title = title.text if title is not None else "Unknown"
                abstract_parts = [at.text for at in article.findall('.//AbstractText') if at.text]
                abstract = " ".join(abstract_parts) or "No abstract."
                authors = []
                for author in article.findall('.//Author'):
                    ln = author.find('LastName')
                    if ln is not None:
                        fn = author.find('ForeName')
                        authors.append(f"{ln.text} {fn.text if fn is not None else ''}")
                year = article.find('.//PubDate/Year')
                results[pmid] = {
                    "pmid": pmid, "title": title, "abstract": abstract,
                    "authors": "; ".join(authors) or "No authors", 
                    "year": year.text if year is not None else None
                }
        except:
            pass
        return results


# ============================================
# INITIALIZE COMPONENTS
# ============================================

logger.info("="*60)
logger.info("PubMed Research Agent v3.2")
logger.info("="*60)

# MedCPT embeddings
medcpt_embeddings = MedCPTEmbeddings(device="cuda", batch_size=8)

# Vectorstore
VECTORSTORE_PATH.mkdir(exist_ok=True)
vectorstore = Chroma(
    persist_directory=str(VECTORSTORE_PATH),
    embedding_function=medcpt_embeddings,
    collection_name="pubmed_papers",
    collection_metadata={"hnsw:space": "cosine"}
)

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Cross-encoder reranker
enc_model = HuggingFaceCrossEncoder(
    model_name='ncbi/MedCPT-Cross-Encoder',
    model_kwargs={'device': 'cuda'}
)
compressor = CrossEncoderReranker(model=enc_model, top_n=10)

# BM25 and Hybrid retriever
bm25_retriever = BM25SparseRetriever(vectorstore)
hybrid_retriever = HybridRetriever(vectorstore, bm25_retriever)

# Async PubMed fetcher
async_fetcher = AsyncPubMedFetcher(email=Entrez.email, api_key=os.getenv("NCBI_API_KEY"))

# Knowledge Graph Manager
kg_manager: Optional[KnowledgeGraphManager] = None

if ENABLE_KNOWLEDGE_GRAPH:
    # ----------------------------------------------------
    # FIX: ROBUST DATABASE CHECK
    # ----------------------------------------------------
    # We check if the DB path exists. We removed .iterdir() so it won't crash 
    # if the OS sees the database as a file instead of a folder.
    
    db_missing = not KUZU_DB_PATH.exists()
    
    # If it exists, we check if it's suspiciously small (e.g. < 1KB), which implies it's empty/corrupt
    is_empty = False
    if KUZU_DB_PATH.exists():
        try:
            # Check size depending on if it's a file or folder
            if KUZU_DB_PATH.is_file():
                is_empty = KUZU_DB_PATH.stat().st_size < 1024
            elif KUZU_DB_PATH.is_dir():
                # If it's a folder, check if it has content
                is_empty = not any(KUZU_DB_PATH.iterdir())
        except Exception:
            # If we can't check, assume it might need reloading
            pass

    if LOAD_PRIMEKG and (db_missing or is_empty):
        logger.info("!!! Database missing or empty. Automatically triggering FAST loader... !!!")
        try:
            # Dynamically import the fast loader script
            import load_primekg
            
            logger.info("--- Step 1: Checking Data ---")
            edges_file = load_primekg.download_primekg()
            
            logger.info("--- Step 2: Preparing Polars CSVs ---")
            entities_csv, relations_csv = load_primekg.prepare_csvs_for_kuzu(edges_file, PRIMEKG_LIMIT)
            
            logger.info("--- Step 3: Bulk Loading (COPY FROM) ---")
            load_primekg.bulk_load_into_kuzu(entities_csv, relations_csv)
            
            logger.info("✓ Fast load complete!")
            
        except ImportError:
            logger.error("Could not import 'load_primekg.py'. Ensure it is in the same folder.")
        except Exception as e:
            logger.error(f"Fast load failed: {e}")
            import traceback
            traceback.print_exc()

    try:
        kg_manager = KnowledgeGraphManager(
            db_path=KUZU_DB_PATH,
            gliner_device=GLINER_DEVICE,
            # We set load_primekg=False here because we handled the loading above (if needed)
            load_primekg=False,  
            primekg_limit=PRIMEKG_LIMIT,
            primekg_data_dir=PRIMEKG_DATA_DIR
        )
    except Exception as e:
        logger.warning(f"KG init failed: {e}")
        import traceback
        traceback.print_exc()
        ENABLE_KNOWLEDGE_GRAPH = False

torch.cuda.empty_cache()
logger.info(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")


# ============================================
# SEARCH FUNCTIONS
# ============================================

def kg_enhanced_search(query, num_results=10):
    """Search with KG enhancement."""
    kg_context = ""
    
    if kg_manager and ENABLE_KG_ENHANCED_RETRIEVAL:
        try:
            expansion = kg_manager.expand_query(query)
            kg_context = kg_manager.retriever.format_kg_context(
                expansion.get('relevant_triples', []), 8
            )
        except Exception as e:
            logger.warning(f"KG expansion failed: {e}")
    
    results = hybrid_retriever.search(query, k=num_results * 2)
    if not results:
        return "No papers found.", kg_context
    
    # Deduplicate
    seen = set()
    unique = [d for d in results if hash(d.page_content[:100]) not in seen 
              and not seen.add(hash(d.page_content[:100]))]
    
    # Rerank
    results = compressor.compress_documents(unique, query)[:num_results] if len(unique) > 1 else unique[:num_results]
    
    # Format output
    papers = {}
    for doc in results:
        pmid = doc.metadata.get("pmid", "unknown")
        if pmid not in papers:
            papers[pmid] = {
                "title": doc.metadata.get("title", "Unknown"),
                "authors": doc.metadata.get("authors", "Unknown"),
                "year": doc.metadata.get("year", "Unknown"),
                "chunks": []
            }
        papers[pmid]["chunks"].append(doc.page_content)
    
    output = [f"Found {len(papers)} paper(s)\n"]
    for idx, (pmid, info) in enumerate(papers.items(), 1):
        content = "\n\n".join(info['chunks'])[:500]
        output.append(f"\n[{idx}] PMID: {pmid}\nTitle: {info['title']}\n"
                     f"Authors: {info['authors']}\nYear: {info['year']}\n"
                     f"{content}...\n{'='*40}")
    
    return "".join(output), kg_context


# ============================================
# ASYNC SEARCH & STORE
# ============================================

async def async_pubmed_search_and_store(keywords, years=None, pnum=5):
    """Search PubMed and store papers."""
    year_list = [str(date.today().year)] if not years else (
        [str(y) for y in range(int(years.split("-")[0]), int(years.split("-")[1])+1)] 
        if "-" in years else [years]
    )
    
    total_stored = 0
    all_papers = []
    
    for year in year_list:
        pmids = await async_fetcher.search(f"({keywords}) AND {year}[pdat]", retmax=pnum)
        if not pmids:
            continue
        
        articles = await async_fetcher.fetch_abstracts(pmids)
        
        for pmid, article in articles.items():
            try:
                content = f"PMID: {pmid}\nTitle: {article['title']}\n" \
                         f"Authors: {article['authors']}\nYear: {year}\n\n{article['abstract']}"
                
                chunks = text_splitter.split_text(content)
                docs = [Document(
                    page_content=c,
                    metadata={
                        "pmid": pmid, "title": article['title'],
                        "authors": article['authors'], "year": year,
                        "chunk_index": i, "source": "pubmed"
                    }
                ) for i, c in enumerate(chunks)]
                
                vectorstore.add_documents(docs)
                
                if kg_manager:
                    try:
                        kg_manager.process_paper(pmid, article['title'], article['abstract'], year)
                    except:
                        pass
                
                all_papers.append(f"{article['title']} (PMID:{pmid})")
                total_stored += 1
            except:
                pass
    
    bm25_retriever.refresh()
    return f"Stored {total_stored} papers:\n" + "\n".join(f"• {p}" for p in all_papers)


def pubmed_search_and_store(keywords, years=None, pnum=10):
    """Sync wrapper for async search."""
    try:
        return asyncio.run(async_pubmed_search_and_store(keywords, years, pnum))
    except:
        try:
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.get_event_loop().run_until_complete(
                async_pubmed_search_and_store(keywords, years, pnum)
            )
        except:
            return "Search failed"


# ============================================
# TOOLS
# ============================================

@tool
def pubmed_search_and_store_tool(keywords: str, years: str = None, pnum: int = 10) -> str:
    """Search PubMed and store papers in the database."""
    return pubmed_search_and_store(keywords, years, pnum)


@tool
def search_rag_database_tool(query: str, num_results: int = 10) -> str:
    """Search the database with KG-enhanced retrieval."""
    results, kg_context = kg_enhanced_search(query, num_results)
    return f"{kg_context}\n\n{results}" if kg_context else results


@tool
def check_rag_for_topic_tool(keywords: str) -> str:
    """Check if papers exist for a topic."""
    results = hybrid_retriever.search(keywords, k=5)
    if not results:
        return f"No papers for '{keywords}'"
    papers = {d.metadata.get("pmid"): d.metadata.get("title") 
              for d in results if d.metadata.get("pmid")}
    return f"Found {len(papers)} papers:\n" + "\n".join(
        f"• PMID:{p} | {t[:50]}..." for p, t in papers.items()
    )


@tool
def get_database_stats_tool() -> str:
    """Get database and KG statistics."""
    output = f"Vector DB: {vectorstore._collection.count()} chunks\n"
    if kg_manager:
        stats = kg_manager.get_stats()
        output += f"KG: {stats.get('entities', 0):,} entities, {stats.get('relations', 0):,} relations"
    return output


@tool
def verify_facts_tool(text: str) -> str:
    """Verify claims in text against the knowledge graph."""
    if not kg_manager:
        return "KG not enabled"
    result = kg_manager.verify_response(text)
    return f"Claims: {result['total_claims']}, Verified: {result['verified_claims']}, " \
           f"Rate: {result['verification_rate']:.0%}"


@tool
def explore_kg_entity_tool(entity_name: str) -> str:
    """Explore an entity in the knowledge graph."""
    if not kg_manager:
        return "KG not enabled"
    
    entities = kg_manager.find_entity(entity_name, limit=3)
    if not entities:
        return f"'{entity_name}' not found"
    
    output = ""
    for e in entities:
        output += f"\n{e['name']} ({e['type']})\n"
        neighbors = kg_manager.get_entity_neighbors(e['id'], limit=5)
        for n in neighbors:
            output += f"  → {n.get('relation', 'related')} → {n['name']}\n"
    return output

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
def gene_census_expression_tool(gene_symbol: str, tissue: str) -> str:
    """
    Query CELLxGENE Census for the TOP 10 cell types expressing a gene in a specific tissue.
    
    OPTIMIZED: Uses incremental queries per cell type instead of loading all cells.
    
    Args:
        gene_symbol: Gene name (e.g., "ACE2", "EGFR")
        tissue: Tissue name (e.g., "lung", "brain") - lowercase recommended
    """
    try:
        import cellxgene_census
        import tiledbsoma as soma
        import numpy as np
        import pandas as pd
    except ImportError:
        return "ERROR: Missing packages. pip install cellxgene-census tiledbsoma"

    if not tissue:
        return "⚠️ ERROR: You MUST provide a 'tissue' parameter."

    # Normalize inputs
    gene_symbol = gene_symbol.strip().upper()
    tissue_normalized = tissue.strip().lower()
    
    output = [f"**Single-Cell Expression: {gene_symbol} in {tissue_normalized}**\n"]

    try:
        with cellxgene_census.open_soma(census_version="2025-11-08") as census:
            human = census["census_data"]["homo_sapiens"]
            
            # ===== Step 1: Resolve Gene ID (fast) =====
            var_df = human.ms["RNA"].var.read(
                value_filter=f"feature_name == '{gene_symbol}'",
                column_names=["soma_joinid", "feature_id", "feature_name"]
            ).concat().to_pandas()
            
            if var_df.empty:
                return f"❌ Gene '{gene_symbol}' not found in Census."
            
            gene_soma_joinid = int(var_df.iloc[0]["soma_joinid"])
            gene_id = var_df.iloc[0]["feature_id"]
            output.append(f"Gene: {gene_id}")

            # ===== Step 2: Get cell types in this tissue (metadata only - FAST) =====
            output.append(f"Fetching cell types in '{tissue_normalized}'...")
            
            obs_df = cellxgene_census.get_obs(
                census, 
                "homo_sapiens",
                value_filter=f"tissue_general == '{tissue_normalized}' and is_primary_data == True",
                column_names=["cell_type"]
            )
            
            if obs_df.empty:
                # Try to suggest correct tissue names
                all_tissues = cellxgene_census.get_obs(
                    census, "homo_sapiens", 
                    column_names=["tissue_general"]
                )["tissue_general"].unique()
                suggestions = [t for t in all_tissues if tissue_normalized in t.lower()][:5]
                return f"❌ Tissue '{tissue_normalized}' not found.\nSuggestions: {suggestions}"
            
            # Count cells per cell type
            cell_type_counts = obs_df["cell_type"].value_counts()
            n_total_cells = len(obs_df)
            
            # Filter to cell types with enough cells (>50 for statistical relevance)
            valid_cell_types = cell_type_counts[cell_type_counts >= 50].index.tolist()
            
            output.append(f"Total cells: {n_total_cells:,}")
            output.append(f"Cell types with ≥50 cells: {len(valid_cell_types)}\n")
            
            del obs_df  # Free memory
            
            # ===== Step 3: Query expression per cell type (OPTIMIZED) =====
            # Use SOMA axis_query for streaming/incremental computation
            
            results = []
            
            for ct in valid_cell_types:
                try:
                    # Build targeted query for this cell type
                    cell_filter = (
                        f"tissue_general == '{tissue_normalized}' and "
                        f"cell_type == '{ct}' and "
                        f"is_primary_data == True"
                    )
                    
                    with human.axis_query(
                        measurement_name="RNA",
                        obs_query=soma.AxisQuery(value_filter=cell_filter),
                        var_query=soma.AxisQuery(coords=(gene_soma_joinid,))  # Single gene!
                    ) as query:
                        
                        # Get cell count
                        n_cells = query.n_obs
                        if n_cells < 50:
                            continue
                        
                        # Compute stats incrementally (streaming)
                        n_expressing = 0
                        total_expr = 0.0
                        
                        for arrow_tbl in query.X("raw").tables():
                            # Arrow table has columns: soma_dim_0, soma_dim_1, soma_data
                            data = arrow_tbl["soma_data"].to_numpy()
                            n_expressing += len(data)  # Non-zero entries
                            total_expr += data.sum()
                        
                        if n_expressing > 0:
                            mean_expr = total_expr / n_cells  # Mean over ALL cells (including zeros)
                            pct_expressing = (n_expressing / n_cells) * 100
                            
                            results.append({
                                "cell_type": ct,
                                "n_cells": n_cells,
                                "n_expressing": n_expressing,
                                "pct_expressing": pct_expressing,
                                "mean_expr": mean_expr
                            })
                
                except Exception as e:
                    logger.warning(f"Error querying {ct}: {e}")
                    continue
            
            # ===== Step 4: Sort and format results =====
            if not results:
                return (f"❌ No significant expression of {gene_symbol} in {tissue_normalized}.\n"
                       f"Checked {len(valid_cell_types)} cell types.")
            
            # Sort by mean expression
            results.sort(key=lambda x: x["mean_expr"], reverse=True)
            top_10 = results[:10]
            
            output.append(f"**Top 10 Expressing Cell Types:**\n")
            for i, r in enumerate(top_10, 1):
                output.append(f"{i}. **{r['cell_type']}**")
                output.append(f"   Mean: {r['mean_expr']:.3f} | "
                            f"{r['pct_expressing']:.1f}% express | "
                            f"n={r['n_cells']:,} ({r['n_expressing']:,} non-zero)")

    except Exception as e:
        import traceback
        return f"❌ Census Error: {str(e)}\n\n{traceback.format_exc()}"

    return "\n".join(output)


@tool
def gene_census_expression_fast(gene_symbol: str, tissue: str) -> str:
    """
    FASTEST version: Uses experimental mean_variance for pre-computed stats.
    
    This is the most efficient approach when you just need mean expression.
    
    Args:
        gene_symbol: Gene name (e.g., "TP53", "EGFR")
        tissue: Tissue name (e.g., "lung", "brain")
    """
    try:
        import cellxgene_census
        import tiledbsoma as soma
        from cellxgene_census.experimental.pp import mean_variance
        import pandas as pd
    except ImportError:
        return "ERROR: Missing packages. pip install cellxgene-census"

    gene_symbol = gene_symbol.strip().upper()
    tissue_normalized = tissue.strip().lower()
    
    output = [f"**Expression of {gene_symbol} in {tissue_normalized}**\n"]

    try:
        with cellxgene_census.open_soma(census_version="2025-11-08") as census:
            human = census["census_data"]["homo_sapiens"]
            
            # Get gene info
            var_df = human.ms["RNA"].var.read(
                value_filter=f"feature_name == '{gene_symbol}'",
                column_names=["soma_joinid", "feature_id"]
            ).concat().to_pandas()
            
            if var_df.empty:
                return f"❌ Gene '{gene_symbol}' not found."
            
            gene_id = var_df.iloc[0]["feature_id"]
            
            # Get unique cell types in tissue
            obs_df = cellxgene_census.get_obs(
                census, "homo_sapiens",
                value_filter=f"tissue_general == '{tissue_normalized}' and is_primary_data == True",
                column_names=["cell_type"]
            )
            
            if obs_df.empty:
                return f"❌ Tissue '{tissue_normalized}' not found."
            
            cell_types = obs_df["cell_type"].value_counts()
            cell_types = cell_types[cell_types >= 50]  # Filter small clusters
            
            output.append(f"Analyzing {len(cell_types)} cell types...\n")
            
            results = []
            
            # Query each cell type using the efficient mean_variance function
            for ct, count in cell_types.items():
                try:
                    cell_filter = (
                        f"tissue_general == '{tissue_normalized}' and "
                        f"cell_type == '{ct}' and "
                        f"is_primary_data == True"
                    )
                    
                    query = human.axis_query(
                        measurement_name="RNA",
                        obs_query=soma.AxisQuery(value_filter=cell_filter),
                        var_query=soma.AxisQuery(value_filter=f"feature_id == '{gene_id}'")
                    )
                    
                    # Use the optimized mean_variance function
                    mv_df = mean_variance(query, axis=0, calculate_mean=True, calculate_variance=False)
                    query.close()
                    
                    if not mv_df.empty:
                        mean_val = mv_df["mean"].iloc[0]
                        if mean_val > 0:
                            results.append({
                                "cell_type": ct,
                                "n_cells": count,
                                "mean_expr": mean_val
                            })
                            
                except Exception as e:
                    continue
            
            if not results:
                return f"❌ No expression found for {gene_symbol} in {tissue_normalized}"
            
            results.sort(key=lambda x: x["mean_expr"], reverse=True)
            
            output.append("**Top 10 by Mean Expression:**\n")
            for i, r in enumerate(results[:10], 1):
                output.append(f"{i}. **{r['cell_type']}**: {r['mean_expr']:.3f} (n={r['n_cells']:,})")

    except Exception as e:
        import traceback
        return f"❌ Error: {str(e)}\n{traceback.format_exc()}"

    return "\n".join(output)

def test_census_connection():
    """Test basic Census connectivity and list available tissues."""
    import cellxgene_census
    
    print("Opening Census...")
    with cellxgene_census.open_soma(census_version="2025-11-08") as census:
        print("✓ Connected!")
        
        # Get census info
        info = census["census_info"]["summary"].read().concat().to_pandas()
        print(f"\nCensus info:\n{info}\n")
        
        human = census["census_data"]["homo_sapiens"]
        
        # List all tissue_general values
        print("Fetching tissue_general values...")
        tissues = human.obs.read(
            column_names=["tissue_general"]
        ).concat().to_pandas()["tissue_general"].value_counts()
        
        print(f"\nAll tissue_general values (count):\n{tissues.to_string()}")
        
        # Test a known gene
        print("\n\nTesting gene lookup for TP53...")
        var_df = human.ms["RNA"].var.read(
            value_filter="feature_name == 'TP53'",
            column_names=["feature_id", "feature_name"]
        ).concat().to_pandas()
        print(f"TP53 lookup:\n{var_df}")

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
    
    For single-cell resolution (specific cell types), use gene_census_expression_tool instead.
    
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
    
    Note: GTEx measures bulk tissue averages. For specific cell type expression
    (e.g., "fibroblasts" or "T-cells"), use gene_census_expression_tool.
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
            "gencodeVersion": "v26",  # GTEx v8 uses Gencode v26
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
            "datasetId": "gtex_v8",
            "format": "json"
        }
        
        headers = {"Accept": "application/json"}
        
        output.append(f"Querying GTEx v8 expression data...")
        
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
        output.append(f"• For specific cell type expression, use gene_census_expression_tool")
        
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
# SCREEN GraphQL HELPER FUNCTION - FIXED
# =========================================

def query_screen_graphql(chrom: str, start: int, end: int, assembly: str = "GRCh38") -> Optional[Dict[str, Any]]:
    """
    Query SCREEN using the new GraphQL API.
    
    FIXED: 
    - Fixed GenomicRegionInput format (chromosome field name)
    - Added 'chr' prefix handling to avoid double prefixing
    - Added better error logging to see actual response
    - Fixed JSON payload structure
    """
    url = "https://screen.api.wenglab.org/graphql"
    
    # FIXED: Don't add 'chr' prefix here - do it in variable construction
    # The chrom parameter might already have it from the tool
    chrom_clean = chrom.replace("chr", "")
    
    query = """
    query cCREs($coordinates: [GenomicRegionInput!]!, $assembly: String!) {
        ccres(coordinates: $coordinates, assembly: $assembly) {
            accession
            group
            dnase
            h3k4me3
            h3k27ac
            ctcf
        }
    }
    """
    
    # FIXED: Ensure proper variable types and structure
    variables = {
        "coordinates": [{
            "chromosome": f"chr{chrom_clean}",  # Add chr prefix here consistently
            "start": int(start),
            "end": int(end)
        }],
        "assembly": assembly.upper()  # Ensure uppercase
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "ResearchAgent/1.0"
    }
    
    payload = {
        "query": query,
        "variables": variables
    }
    
    try:
        # Log what we're sending for debugging
        logger.debug(f"SCREEN GraphQL Request: {json.dumps(payload, indent=2)}")
        
        r = requests.post(
            url, 
            json=payload,
            headers=headers, 
            timeout=REQUEST_TIMEOUT
        )
        
        # Log the response for debugging
        logger.debug(f"SCREEN GraphQL Response Status: {r.status_code}")
        if not r.ok:
            logger.error(f"SCREEN GraphQL HTTP {r.status_code}: {r.text[:500]}")
            return None
            
        data = r.json()
        
        # Check for GraphQL-level errors
        if "errors" in data:
            error_msgs = [e.get("message", "Unknown") for e in data["errors"]]
            logger.error(f"SCREEN GraphQL Errors: {error_msgs}")
            # Return data anyway if partial results exist
            return data if "data" in data else None
            
        return data
        
    except requests.exceptions.Timeout:
        logger.error("SCREEN GraphQL request timed out")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"SCREEN GraphQL request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"SCREEN GraphQL invalid JSON: {e}. Response: {r.text[:200]}")
        return None


def query_screen_ccre_details(accession: str, assembly: str = "GRCh38") -> Optional[Dict[str, Any]]:
    """
    Query detailed cCRE information including tissue-specific activity.
    
    FIXED: Better error handling and logging
    """
    url = "https://screen.api.wenglab.org/graphql"
    
    query = """
    query cCREDetails($accession: String!, $assembly: String!) {
        ccre(accession: $accession, assembly: $assembly) {
            accession
            group
            coordinates {
                chromosome
                start
                end
            }
            ctspecific {
                biosample_summary
                dnase_zscore
                h3k4me3_zscore
                h3k27ac_zscore
                ctcf_zscore
            }
        }
    }
    """
    
    variables = {
        "accession": accession,
        "assembly": assembly.upper()
    }
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "ResearchAgent/1.0"
    }
    
    payload = {
        "query": query,
        "variables": variables
    }
    
    try:
        logger.debug(f"SCREEN cCRE Details Request: {json.dumps(payload)}")
        
        r = requests.post(
            url, 
            json=payload,
            headers=headers, 
            timeout=REQUEST_TIMEOUT
        )
        
        if not r.ok:
            logger.error(f"SCREEN cCRE details HTTP {r.status_code}: {r.text[:500]}")
            return None
            
        data = r.json()
        
        if "errors" in data:
            error_msgs = [e.get("message", "Unknown") for e in data["errors"]]
            logger.error(f"SCREEN cCRE details GraphQL Errors: {error_msgs}")
            return data if "data" in data else None
            
        return data
        
    except requests.exceptions.Timeout:
        logger.error("SCREEN cCRE details request timed out")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"SCREEN cCRE details request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"SCREEN cCRE details invalid JSON: {e}")
        return None
# =========================================
# REGULATORY TISSUE ACTIVITY TOOL
# FIXED: Uses new GraphQL API with REST API fallback
# =========================================

@tool
def regulatory_tissue_activity_tool(chromosome: str, start: int, end: int, tissue: str) -> str:
    """
    Check if a genomic region has active regulatory elements in a specific tissue.
    
    Uses ENCODE SCREEN database to find candidate cis-Regulatory Elements (cCREs)
    and their activity (DNase/H3K27ac Z-scores) in tissue-matched biosamples.
    
    Args:
        chromosome: Chromosome number (e.g., "17" or "chr17")
        start: Start position (GRCh38)
        end: End position (GRCh38)
        tissue: Tissue name (e.g., "liver", "lung", "brain")
    
    Returns:
        Regulatory activity status with DATA SOURCE STATUS
    """
    chrom = chromosome.replace("chr", "")
    
    output = [f"**Regulatory Activity Analysis**"]
    output.append(f"Region: chr{chrom}:{start:,}-{end:,}")
    output.append(f"Tissue: {tissue}\n")
    
    sources_status = {
        "ENCODE SCREEN (GraphQL)": "❌ NOT QUERIED",
        "ENCODE SCREEN Details": "❌ NOT QUERIED"
    }
    any_real_data = False
    
    # =========================================
    # Try new GraphQL API first
    # =========================================
    try:
        data = query_screen_graphql(chrom, start, end)
        
        # FIXED: Check for None return
        if data is None:
            sources_status["ENCODE SCREEN (GraphQL)"] = "❌ REQUEST FAILED"
            output.append("❌ Failed to query SCREEN GraphQL API (timeout or connection error)")
            
        elif "errors" in data and "data" not in data:
            # GraphQL error with no data
            error_msg = data.get("errors", [{}])[0].get("message", "Unknown error")[:100]
            sources_status["ENCODE SCREEN (GraphQL)"] = f"❌ GraphQL ERROR - {error_msg}"
            output.append(f"⚠️ SCREEN GraphQL error: {error_msg}")
            
        else:
            ccres = data.get("data", {}).get("ccres", [])
            
            if not ccres:
                sources_status["ENCODE SCREEN (GraphQL)"] = "✅ SUCCESS (0 cCREs found)"
                output.append("❌ **No regulatory elements found at this location.**")
                output.append("This region may not have annotated cCREs in ENCODE SCREEN.")
            else:
                any_real_data = True
                sources_status["ENCODE SCREEN (GraphQL)"] = f"✅ SUCCESS ({len(ccres)} cCREs)"
                output.append(f"Found **{len(ccres)} cCRE(s)** at this location:\n")
                
                tissue_lower = tissue.lower()
                details_checked = 0
                details_failed = 0
                
                for ccre in ccres[:5]:
                    accession = ccre.get("accession", "Unknown")
                    element_type = ccre.get("group", "Unknown")
                    
                    output.append(f"**{accession}** ({element_type})")
                    
                    # Get detailed tissue-specific information
                    detail_data = query_screen_ccre_details(accession)
                    
                    # FIXED: Check for None return
                    if detail_data is None:
                        details_failed += 1
                        output.append(f"  ⚠️ Failed to fetch details (connection error)")
                        continue
                    
                    if "errors" in detail_data:
                        details_failed += 1
                        output.append(f"  ⚠️ Details fetch failed (GraphQL error)")
                        continue
                    
                    ccre_detail = detail_data.get("data", {}).get("ccre", {})
                    activity_data = ccre_detail.get("ctspecific", []) or []
                    
                    if activity_data:
                        details_checked += 1
                        matches = []
                        
                        for entry in activity_data:
                            # FIXED: Safe string handling
                            bio_name = str(entry.get("biosample_summary") or "").lower()
                            
                            if tissue_lower in bio_name:
                                z_score = entry.get("dnase_zscore") or entry.get("h3k27ac_zscore", 0)
                                # FIXED: Safe numeric comparison
                                if z_score is not None and float(z_score) > 1.64:
                                    sample_name = entry.get("biosample_summary", "Unknown")
                                    matches.append(f"{sample_name} (Z={float(z_score):.1f})")
                        
                        if matches:
                            output.append(f"  ✅ **ACTIVE in {tissue}**")
                            output.append(f"     Evidence: {', '.join(matches[:3])}")
                        else:
                            output.append(f"  ❌ No significant activity in {tissue}")
                    else:
                        output.append(f"  ⚠️ No tissue-specific data available")
                    
                    output.append("")
                
                if details_checked > 0:
                    sources_status["ENCODE SCREEN Details"] = f"✅ {details_checked} checked, {details_failed} failed"
                else:
                    sources_status["ENCODE SCREEN Details"] = "❌ ALL FAILED"
                    
    except Exception as e:
        error_msg = str(e)[:100]
        sources_status["ENCODE SCREEN (GraphQL)"] = f"❌ ERROR: {error_msg}"
        output.append(f"Error: {error_msg}")
        logger.exception("Unexpected error in regulatory_tissue_activity_tool")
    
    # =========================================
    # DATA SOURCE STATUS SUMMARY
    # =========================================
    output.append("\n" + "="*50)
    output.append("**DATA SOURCE STATUS:**")
    for source, status in sources_status.items():
        output.append(f"  • {source}: {status}")
    
    if not any_real_data:
        output.append("\n❌ **NO REGULATORY DATA RETRIEVED**")
        output.append("⚠️ DO NOT make claims about regulatory activity without verified data.")
    
    return "\n".join(output)

# =========================================
# TE TISSUE ACTIVITY TOOL
# FIXED: Uses new GraphQL API
# =========================================
@tool  
def te_tissue_activity_tool(chromosome: str, start: int, end: int, tissue: str) -> str:
    """
    Check if a Transposable Element (TE) region shows transcriptional activity.
    
    This provides INDIRECT evidence only. For definitive TE quantification,
    use specialized tools like TEtranscripts or SQuIRE on raw RNA-seq data.
    
    Args:
        chromosome: Chromosome (e.g., "17" or "chr17")
        start: TE start position (GRCh38)
        end: TE end position (GRCh38)
        tissue: Tissue name (e.g., "liver", "placenta")
    
    Returns:
        Evidence summary with DATA SOURCE STATUS
    """
    chrom = chromosome.replace("chr", "")
    
    output = [f"**Transposable Element Activity Analysis**"]
    output.append(f"Region: chr{chrom}:{start:,}-{end:,}")
    output.append(f"Tissue: {tissue}\n")
    
    sources_status = {
        "ENCODE SCREEN (chromatin)": "❌ NOT QUERIED",
        "ENCODE RNA-seq search": "❌ NOT QUERIED",
        "ENCODE H3K36me3 search": "❌ NOT QUERIED"
    }
    evidence_score = 0
    
    # 1. Chromatin state using GraphQL
    output.append("**1. Chromatin State:**")
    try:
        data = query_screen_graphql(chrom, start, end)
        
        # FIXED: Check for None return
        if data is None:
            sources_status["ENCODE SCREEN (chromatin)"] = "❌ REQUEST FAILED"
            output.append("   ⚠️ Failed to query SCREEN API (timeout or connection error)")
            
        elif "errors" in data and "data" not in data:
            sources_status["ENCODE SCREEN (chromatin)"] = "❌ GraphQL ERROR"
            output.append(f"   ⚠️ SCREEN API error: {data['errors'][0].get('message', 'Unknown')[:100]}")
            
        else:
            ccres = data.get("data", {}).get("ccres", [])
            
            if ccres:
                sources_status["ENCODE SCREEN (chromatin)"] = f"✅ SUCCESS ({len(ccres)} elements)"
                output.append(f"   Found {len(ccres)} regulatory element(s) overlapping this TE")
                
                for ccre in ccres[:2]:
                    acc = ccre.get("accession")
                    if not acc:
                        continue
                        
                    detail_data = query_screen_ccre_details(acc)
                    
                    # FIXED: Check for None
                    if detail_data is None:
                        continue
                        
                    if "errors" in detail_data:
                        continue
                        
                    ccre_detail = detail_data.get("data", {}).get("ccre", {})
                    activity = ccre_detail.get("ctspecific", []) or []
                    
                    if activity:
                        tissue_lower = tissue.lower()
                        for entry in activity:
                            # FIXED: Safe string handling
                            bio_name = str(entry.get("biosample_summary") or "").lower()
                            
                            if tissue_lower in bio_name:
                                z = entry.get("dnase_zscore", 0)
                                # FIXED: Safe numeric comparison
                                if z is not None and float(z) > 1.64:
                                    output.append(f"   ✅ Open chromatin in {tissue} (Z={float(z):.1f})")
                                    evidence_score += 2
                                    break
            else:
                sources_status["ENCODE SCREEN (chromatin)"] = "✅ SUCCESS (0 elements - closed chromatin)"
                output.append("   ❌ No regulatory elements at this location (closed chromatin)")
                
    except requests.exceptions.Timeout:
        sources_status["ENCODE SCREEN (chromatin)"] = "❌ TIMEOUT"
        output.append("   ⚠️ SCREEN API timeout")
    except Exception as e:
        sources_status["ENCODE SCREEN (chromatin)"] = "❌ ERROR"
        output.append(f"   ⚠️ Error: {str(e)[:50]}")
        logger.exception("Error in TE activity chromatin check")
    
    # 2. RNA-seq datasets
    output.append("\n**2. Transcription Evidence (RNA-seq):**")
    try:
        encode_url = "https://www.encodeproject.org/search/"
        params = {
            "type": "Experiment",
            "assay_term_name": "polyA plus RNA-seq",
            "biosample_ontology.term_name": tissue,
            "status": "released",
            "format": "json",
            "limit": 5
        }
        headers = {
            "Accept": "application/json",
            "User-Agent": "ResearchAgent/1.0 (research@example.com)"  # FIXED: Added User-Agent
        }
        
        r = requests.get(encode_url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        
        if r.ok:
            data = r.json()
            experiments = data.get("@graph", [])
            
            if experiments:
                sources_status["ENCODE RNA-seq search"] = f"✅ SUCCESS ({len(experiments)} datasets)"
                output.append(f"   Found {len(experiments)} RNA-seq datasets for {tissue}")
                output.append(f"   (Manual inspection needed to check TE expression)")
                evidence_score += 1
            else:
                sources_status["ENCODE RNA-seq search"] = "⚠️ NO DATASETS"
                output.append(f"   No RNA-seq data available for {tissue}")
        else:
            sources_status["ENCODE RNA-seq search"] = f"❌ HTTP {r.status_code}"
            output.append(f"   ⚠️ ENCODE API error: HTTP {r.status_code}")
            
    except requests.exceptions.Timeout:
        sources_status["ENCODE RNA-seq search"] = "❌ TIMEOUT"
        output.append("   ⚠️ Timeout")
    except Exception as e:
        sources_status["ENCODE RNA-seq search"] = f"❌ ERROR"
        output.append(f"   ⚠️ Error: {str(e)[:50]}")
        logger.exception("Error in TE activity RNA-seq check")
    
    # 3. H3K36me3 (transcription mark)
    output.append("\n**3. Elongation Mark (H3K36me3):**")
    try:
        encode_url = "https://www.encodeproject.org/search/"
        params = {
            "type": "Experiment",
            "assay_term_name": "Histone ChIP-seq",
            "target.label": "H3K36me3",
            "biosample_ontology.term_name": tissue,
            "status": "released",
            "format": "json",
            "limit": 3
        }
        headers = {
            "Accept": "application/json",
            "User-Agent": "ResearchAgent/1.0 (research@example.com)"  # FIXED: Added User-Agent
        }
        
        r = requests.get(encode_url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        
        if r.ok:
            data = r.json()
            experiments = data.get("@graph", [])
            
            if experiments:
                sources_status["ENCODE H3K36me3 search"] = f"✅ SUCCESS ({len(experiments)} datasets)"
                output.append(f"   Found {len(experiments)} H3K36me3 datasets for {tissue}")
                output.append(f"   (Peak overlap would indicate active transcription)")
                evidence_score += 1
            else:
                sources_status["ENCODE H3K36me3 search"] = "⚠️ NO DATASETS"
                output.append(f"   No H3K36me3 data for {tissue}")
        else:
            sources_status["ENCODE H3K36me3 search"] = f"❌ HTTP {r.status_code}"
                
    except requests.exceptions.Timeout:
        sources_status["ENCODE H3K36me3 search"] = "❌ TIMEOUT"
    except Exception as e:
        sources_status["ENCODE H3K36me3 search"] = f"❌ ERROR"
        logger.exception("Error in TE activity H3K36me3 check")
    
    # Summary
    output.append("\n" + "="*50)
    
    if evidence_score >= 3:
        output.append(f"**ASSESSMENT: LIKELY ACTIVE** (evidence score: {evidence_score})")
    elif evidence_score >= 1:
        output.append(f"**ASSESSMENT: POSSIBLY ACTIVE** (evidence score: {evidence_score})")
    else:
        output.append(f"**ASSESSMENT: LIKELY SILENT** (evidence score: {evidence_score})")
    
    output.append("\n⚠️ **NOTE:** This is INDIRECT evidence only.")
    output.append("For definitive TE expression: use TEtranscripts or SQuIRE on raw RNA-seq.")
    
    # DATA SOURCE STATUS
    output.append("\n" + "="*50)
    output.append("**DATA SOURCE STATUS:**")
    for source, status in sources_status.items():
        output.append(f"  • {source}: {status}")
    
    # FIXED: Better failure detection
    all_failed = all(
        "❌" in status or "TIMEOUT" in status or "SKIPPED" in status 
        for status in sources_status.values()
    )
    if all_failed:
        output.append("\n❌ **ALL DATA SOURCES FAILED**")
        output.append("⚠️ DO NOT make claims about TE activity without verified data.")
    
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
            output.append(f"  Start: {start:,}")
            output.append(f"  End: {end:,}")
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
1. **NO HALLUCINATION:** Never guess gene functions, expression levels, or paper citations. If a tool returns no data, state "No data found."
2. **VERIFY FIRST:** You must verify a gene's identity (using `gene_info_tool`) before discussing its function or expression.
3. **CITE SOURCES:** Only cite PMIDs or data sources (e.g., "GTEx v8", "CELLxGENE Census") that explicitly appear in tool outputs.

# TOOL ROUTING GUIDE (How to choose the right tool)

## CATEGORY 1: GENE QUESTIONS ("What is GENE?", "Where is GENE expressed?")
* **Step 1: Identity (MANDATORY)**
    * Use `gene_info_tool(gene_symbol)`
    * *Goal:* Get the official symbol, summary, and aliases.
* **Step 2: Expression (If asked about tissues/cells)**
    * **Context: "Overall / Bulk tissue"** (e.g., "Is EGFR in the lung?")
        * Use `gene_tissue_expression_tool(gene_symbol, tissue)`
        * *Source:* GTEx (Bulk RNA-seq). Measures average expression in whole tissue samples.
    * **Context: "Specific Cells / Rare types"** (e.g., "EGFR in lung fibroblasts?")
        * Use `gene_census_expression_tool(gene_symbol, tissue, cell_type)`
        * *Source:* CELLxGENE Census (Single-cell RNA-seq). Measures expression in specific cell subpopulations.
* **Step 3: Coordinates (If asked about location)**
    * Use `get_gene_coordinates_tool(gene_symbol)`

## CATEGORY 2: REGULATION & EPIGENETICS ("Is this region active?", "Enhancers in brain?")
* **Step 1: Define Region**
    * If user gives a gene name, first get coordinates via `get_gene_coordinates_tool`.
    * If user gives coordinates (chr:start-end), use them directly.
* **Step 2: Check Regulatory Elements**
    * Use `regulatory_tissue_activity_tool(chrom, start, end, tissue)`
    * *Source:* ENCODE SCREEN. Checks for cCREs (Enhancers/Promoters) active in that tissue.
* **Step 3: Check Transposable Elements (TEs)**
    * Use `te_tissue_activity_tool(chrom, start, end, tissue)`
    * *Source:* ENCODE RNA-seq/ChIP-seq. Checks if a TE is transcribed.

## CATEGORY 3: LITERATURE REVIEW ("Find papers on...", "Summarize studies...")
* **Step 1: Check Existing Knowledge**
    * Use `check_rag_for_topic_tool(keywords)` to see if we already have papers.
* **Step 2: Search External (If needed)**
    * Use `pubmed_search_and_store_tool(keywords, years, pnum)` to fetch new papers.
* **Step 3: Synthesize**
    * Use `search_rag_database_tool(query)` to answer using the stored papers and Knowledge Graph.

# RESPONSE FORMATTING
* **Gene Function:** Start with the official summary from `gene_info_tool`.
* **Expression Data:**
    * Clearly distinguish **Bulk** (GTEx) from **Single-Cell** (Census).
    * Report the units (TPM for bulk, % expressing cells for single-cell).
    * *Example:* "In bulk lung tissue, EGFR expression is moderate (Median TPM: 15.2). However, single-cell analysis shows it is highly specific to Basal Cells (85% expressing)."
* **Citations:** Use standard format `[PMID: 12345678]`.

# EXECUTION LOOP
1. **Analyze Request:** Identify the biological entities (Genes, Tissues, Diseases).
2. **Select Tool:** Pick the tool from the Routing Guide above.
3. **Observe Output:** Read the tool's raw output.
4. **Refine/Answer:** If tool fails (e.g., "Gene not found"), try an alias or report the error. If successful, synthesize the answer.
"""

# system_prompt = """You are a biomedical research assistant with access to PubMed literature, a biomedical knowledge graph, and genomic expression databases.

# ## ⚠️ CRITICAL RULE: VERIFY GENES BEFORE ANSWERING

# **ALWAYS use `gene_info_tool` FIRST before answering ANY question about a gene.**

# This prevents hallucination. Do NOT guess what a gene does based on its name.

# Example - WRONG approach:
#   User: "What does EGFLAM do?"
#   Assistant: "EGFLAM is related to EGF signaling..." ← HALLUCINATION!

# Example - CORRECT approach:
#   User: "What does EGFLAM do?"
#   Assistant: [calls gene_info_tool("EGFLAM")]
#   Assistant: "According to NCBI Gene, EGFLAM encodes EGF-like, fibronectin type III and laminin G domains protein..."

# ## AVAILABLE TOOLS

# ### Gene Information (USE FIRST!)

# 1. **gene_info_tool**(gene_symbol) ⭐ ALWAYS USE FIRST FOR GENE QUESTIONS
#    - Get authoritative gene information from NCBI Gene and UniProt
#    - Returns: Official name, function, aliases, disease associations
#    - **MANDATORY** before answering questions about what a gene does
#    - Example: `gene_symbol="EGFLAM"` → official function from databases

# ### Literature & Knowledge Graph Tools

# 2. **pubmed_search_and_store_tool**(keywords, years, pnum)
#    - Search PubMed and store papers in the local database
#    - Example: `keywords="EGFR lung cancer therapy", years="2022-2024", pnum=10`

# 3. **search_rag_database_tool**(query, num_results)
#    - Search stored papers with KG-enhanced retrieval
#    - Returns relevant chunks with PMID citations

# 4. **check_rag_for_topic_tool**(keywords)
#    - Quick check if papers exist for a topic before searching

# 5. **get_database_stats_tool**()
#    - Get database and knowledge graph statistics

# 6. **verify_facts_tool**(text)
#    - Verify claims against the knowledge graph

# 7. **explore_kg_entity_tool**(entity_name)
#    - Explore an entity's relationships in the knowledge graph

# ### Genomic Expression Tools

# 8. **gene_tissue_expression_tool**(gene_symbol, tissue=None)
#    - Get gene expression data across tissues
#    - Sources: GTEx (bulk RNA-seq TPM)
#    - Example: `gene_symbol="EGFR", tissue="Lung"`

# 9. **regulatory_tissue_activity_tool**(chromosome, start, end, tissue)
#    - Check if a genomic region has active regulatory elements
#    - Uses ENCODE SCREEN database for cCRE activity

# 10. **te_tissue_activity_tool**(chromosome, start, end, tissue)
#     - Check if a Transposable Element shows activity in a tissue

# 11. **get_gene_coordinates_tool**(gene_symbol)
#     - Get genomic coordinates (chr, start, end) for a gene

# ## WORKFLOW FOR GENE QUESTIONS

# ### Step 1: ALWAYS verify the gene first
# ```
# User: "What does GENEX do?"
# → Call gene_info_tool("GENEX")
# → Read the official function from NCBI/UniProt
# → ONLY THEN answer based on the tool output
# ```

# ### Step 2: If asked about expression, call the expression tools
# ```
# → Call gene_tissue_expression_tool("GENEX")
# → Report actual TPM values from GTEx

# ```

# ### Step 3: If gene not found
# ```
# gene_info_tool returns "NO AUTHORITATIVE DATA FOUND"
# → Tell the user: "I could not find information about this gene in NCBI Gene or UniProt databases."
# → Do NOT guess or make up information!
# ```

# ## RULES

# 1. **Gene Verification**: ALWAYS call `gene_info_tool` before making claims about ANY gene. If the tool says "NO AUTHORITATIVE DATA FOUND", admit you cannot find information - do NOT guess.

# 2. **Citations**: Only cite PMIDs from tool results. Never invent citations.

# 3. **Expression Data**: 
#    - GTEx values are in TPM (Transcripts Per Million)
#    - Always mention the data source
#    - Don't claim expression without calling the expression tool

# 4. **Regulatory Activity**: Z-score > 1.64 = significant (p < 0.05)

# 5. **TE Activity**: Recommend validation with TEtranscripts/SQuIRE for definitive answers

# 6. **Uncertainty**: If tools fail or return no data, say so clearly. Never guess.

# 7. **Coordinates**: All genomic coordinates use GRCh38/hg38.
# """

# system_prompt = """You are a PubMed research assistant with Knowledge Graph integration.

# TOOLS:
# 1. pubmed_search_and_store_tool - Search PubMed and store papers
# 2. search_rag_database_tool - Search stored papers with KG enhancement  
# 3. check_rag_for_topic_tool - Check if papers exist for a topic
# 4. get_database_stats_tool - Get database statistics
# 5. verify_facts_tool - Verify claims against knowledge graph
# 6. explore_kg_entity_tool - Explore entities in knowledge graph

# RULES:
# - Only cite PMIDs from tool results
# - Use KG context when provided
# - Never invent facts or citations
# """


tools = [
    pubmed_search_and_store_tool,
    search_rag_database_tool,
    check_rag_for_topic_tool,
    get_database_stats_tool,
    verify_facts_tool,
    explore_kg_entity_tool,
    gene_tissue_expression_tool,
    regulatory_tissue_activity_tool,
    te_tissue_activity_tool,
    get_gene_coordinates_tool,
    gene_info_tool,
    gene_census_expression_tool
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
    print("PubMed Research Agent v3.2 - Knowledge Graph Edition")
    print("="*60)
    print("Commands: exit, stats, kg, vram, gc, verify <text>, entity <n>")
    print()
    print(get_database_stats_tool.invoke({}))
    print()
    
    while True:
        try:
            user_input = input(">>> ").strip()
            
            if user_input.lower() in {"exit", "quit"}:
                if kg_manager:
                    kg_manager.close()
                print("Goodbye!")
                break
            
            if user_input.lower() == "stats":
                print(f"\n{get_database_stats_tool.invoke({})}\n")
                continue
            
            if user_input.lower() == "kg":
                if kg_manager:
                    print(f"\nKG: {kg_manager.get_stats()}\n")
                continue
            
            if user_input.lower() == "vram":
                alloc = torch.cuda.memory_allocated() / 1e9
                total = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"\nVRAM: {alloc:.2f} / {total:.2f} GB\n")
                continue
            
            if user_input.lower() == "gc":
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                if kg_manager:
                    kg_manager.unload_gliner()
                print("Memory cleared\n")
                continue
            
            if user_input.lower().startswith("verify "):
                print(f"\n{verify_facts_tool.invoke({'text': user_input[7:]})}\n")
                continue
            
            if user_input.lower().startswith("entity "):
                print(f"\n{explore_kg_entity_tool.invoke({'entity_name': user_input[7:]})}\n")
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
            if kg_manager:
                kg_manager.close()
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

