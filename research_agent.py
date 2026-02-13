"""
PubMed Research Agent v3.3 (with Dedup & Full-Text Integration)
===============================================================
Optimized for 8B parameter models (Ministral 3 8B).

Changes from v3.2:
- Merged check_rag + search_rag into single `search_literature_tool`
- All tool docstrings compressed to 1 line (prevents hallucination from extra tokens)
- System prompt restructured as keyword-routing table
- PMID hallucination guard: RAG tool explicitly flags when no papers match
- Tool count reduced from 10 to 9

Integrated from v4.0-lite:
- [IMPROVEMENT 1] PMID dedup at ingest — prevents duplicate chunks in vectorstore
- [IMPROVEMENT 2] Europe PMC full-text fetching (PMID→PMCID conversion)
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
from typing import Tuple, List, Optional, Dict, Any
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
from pydantic import BaseModel, Field
import logging
import requests
import numpy as np

# NEW: Import AlphaGenome (Wrap in try/except to prevent crash if not installed)
try:
    from alphagenome.models import dna_client
    from alphagenome.data import genome
    ALPHAGENOME_AVAILABLE = True
except ImportError:
    ALPHAGENOME_AVAILABLE = False

# Import knowledge graph module
from knowledge_graph import KnowledgeGraphManager
from alphagenome_tool import AlphaGenomePredictor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if not ALPHAGENOME_AVAILABLE:
    logger.warning("AlphaGenome library not found. Tool will be disabled.")

# Constants
REQUEST_TIMEOUT = 15

load_dotenv()
ALPHAGENOME_API_KEY = os.getenv("ALPHAGENOME_API_KEY")
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
    
    # =============================================
    # [IMPROVEMENT 2] EUROPE PMC FULL-TEXT FETCHING
    # =============================================
    
    async def _pmid_to_pmcid(self, session, pmid: str) -> Optional[str]:
        """Convert PMID to PMCID via NCBI ID Converter (free, no key)."""
        url = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
        params = {"ids": pmid, "format": "json"}
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as r:
                if r.status == 200:
                    data = await r.json(content_type=None)
                    records = data.get("records", [])
                    if records and records[0].get("pmcid"):
                        return records[0]["pmcid"]
        except Exception:
            pass
        return None
    
    async def fetch_full_text_europmc(self, session, pmid: str) -> Optional[str]:
        """Try to get full text from Europe PMC (free, no key, open-access papers)."""
        try:
            # Step 1: Convert PMID to PMCID
            async with self.semaphore:
                pmcid = await self._pmid_to_pmcid(session, pmid)
                await asyncio.sleep(0.2)
            
            if not pmcid:
                return None  # Paper not in PMC (not open-access)
            
            # Step 2: Fetch full-text XML using PMCID
            url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
            async with self.semaphore:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
                    if r.status == 200:
                        xml_text = await r.text()
                        root = ET.fromstring(xml_text)
                        body = root.find('.//body')
                        if body is not None:
                            paragraphs = []
                            for p in body.iter('p'):
                                text = ''.join(p.itertext()).strip()
                                if text and len(text) > 20:
                                    paragraphs.append(text)
                            if paragraphs:
                                full_text = "\n\n".join(paragraphs)
                                # Cap at ~8000 chars to keep chunks manageable
                                return full_text[:8000]
                    await asyncio.sleep(0.2)
        except Exception as e:
            logger.debug(f"Europe PMC full-text not available for PMID:{pmid}: {e}")
        return None
    
    async def fetch_full_texts(self, pmids: List[str]) -> Dict[str, Optional[str]]:
        """Try to fetch full texts for a list of PMIDs from Europe PMC."""
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        results = {}
        async with aiohttp.ClientSession() as session:
            for pmid in pmids:
                full_text = await self.fetch_full_text_europmc(session, pmid)
                results[pmid] = full_text
                if full_text:
                    logger.info(f"  ✓ Full-text retrieved for PMID:{pmid} ({len(full_text)} chars)")
        return results
    
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
logger.info("PubMed Research Agent v3.3 (with Dedup & Full-Text)")
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
    db_missing = not KUZU_DB_PATH.exists()
    
    is_empty = False
    if KUZU_DB_PATH.exists():
        try:
            if KUZU_DB_PATH.is_file():
                is_empty = KUZU_DB_PATH.stat().st_size < 1024
            elif KUZU_DB_PATH.is_dir():
                is_empty = not any(KUZU_DB_PATH.iterdir())
        except Exception:
            pass

    if LOAD_PRIMEKG and (db_missing or is_empty):
        logger.info("!!! Database missing or empty. Automatically triggering FAST loader... !!!")
        try:
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
# [IMPROVEMENT 1] PMID DEDUP HELPER
# ============================================

def is_pmid_stored(pmid: str) -> bool:
    """Check if a PMID is already in the vectorstore."""
    try:
        existing = vectorstore._collection.get(
            where={"pmid": pmid},
            limit=1
        )
        return bool(existing and existing.get('ids'))
    except Exception:
        return False


# ============================================
# SEARCH FUNCTIONS
# ============================================

def kg_enhanced_search(query, num_results=10):
    """Search with KG enhancement. Returns (results_text, kg_context, has_relevant_papers)."""
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
        return "NO_PAPERS_FOUND", kg_context, False
    
    # Deduplicate
    seen = set()
    unique = [d for d in results if hash(d.page_content[:100]) not in seen 
              and not seen.add(hash(d.page_content[:100]))]
    
    # Rerank
    results = compressor.compress_documents(unique, query)[:num_results] if len(unique) > 1 else unique[:num_results]
    
    # ====================================================
    # RELEVANCE CHECK: Determine if results actually match
    # ====================================================
    # Extract query keywords for relevance scoring
    query_keywords = set(re.findall(r'\w+', query.lower()))
    # Remove common stop words
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'of', 'in', 'to', 'for',
                  'and', 'or', 'on', 'with', 'by', 'from', 'what', 'how', 'which', 'about',
                  'does', 'do', 'can', 'role', 'effect', 'between', 'this', 'that', 'it'}
    query_keywords -= stop_words
    
    relevant_paper_count = 0
    for doc in results:
        content_lower = doc.page_content.lower()
        matched = sum(1 for kw in query_keywords if kw in content_lower)
        # A paper is "relevant" if it matches at least 40% of query keywords
        if query_keywords and (matched / len(query_keywords)) >= 0.4:
            relevant_paper_count += 1
    
    has_relevant = relevant_paper_count > 0
    
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
    
    return "".join(output), kg_context, has_relevant


# ============================================
# ASYNC SEARCH & STORE
# ============================================

async def async_pubmed_search_and_store(keywords, years=None, pnum=5):
    """Search PubMed, fetch full-text where available, store with dedup."""
    year_list = [str(date.today().year)] if not years else (
        [str(y) for y in range(int(years.split("-")[0]), int(years.split("-")[1])+1)] 
        if "-" in years else [years]
    )
    
    total_stored = 0
    total_skipped = 0
    total_fulltext = 0
    all_papers = []
    
    # STEP 1: PubMed Search
    pubmed_articles = {}
    for year in year_list:
        pmids = await async_fetcher.search(f"({keywords}) AND {year}[pdat]", retmax=pnum)
        if pmids:
            articles = await async_fetcher.fetch_abstracts(pmids)
            pubmed_articles.update(articles)
    
    logger.info(f"PubMed: found {len(pubmed_articles)} papers")
    
    if not pubmed_articles:
        return "No papers found on PubMed for this query."
    
    # STEP 2: Fetch Full-Texts from Europe PMC
    all_pmids = list(pubmed_articles.keys())
    logger.info(f"Attempting full-text fetch for {len(all_pmids)} PMIDs via Europe PMC...")
    full_texts = await async_fetcher.fetch_full_texts(all_pmids)
    ft_count = sum(1 for v in full_texts.values() if v)
    logger.info(f"  Full-text retrieved for {ft_count}/{len(all_pmids)} papers")
    
    # STEP 3: Store papers (with dedup + full-text)
    for pmid, article in pubmed_articles.items():
        if is_pmid_stored(pmid):
            logger.debug(f"Skipping duplicate: {pmid}")
            total_skipped += 1
            continue
        
        try:
            # Use full-text if available, otherwise abstract
            text_content = full_texts.get(pmid) or article['abstract']
            has_fulltext = full_texts.get(pmid) is not None
            if has_fulltext:
                total_fulltext += 1
            
            year = article.get('year') or year_list[0]
            source = "pubmed+fulltext" if has_fulltext else "pubmed"
            
            content = f"PMID: {pmid}\nTitle: {article['title']}\n" \
                     f"Authors: {article['authors']}\nYear: {year}\n\n{text_content}"
            
            chunks = text_splitter.split_text(content)
            docs = [Document(
                page_content=c,
                metadata={
                    "pmid": pmid, "title": article['title'],
                    "authors": article['authors'], "year": year,
                    "chunk_index": i, "source": source
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
    
    summary_parts = [f"Stored {total_stored} papers"]
    if total_skipped > 0:
        summary_parts.append(f"({total_skipped} duplicates skipped)")
    if total_fulltext > 0:
        summary_parts.append(f"({total_fulltext} with full-text)")
    
    summary = " ".join(summary_parts) + ":\n"
    summary += "\n".join(f"• {p}" for p in all_papers)
    return summary


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
# TOOLS (1-line docstrings for 8B model)
# ============================================

@tool
def pubmed_search_and_store_tool(keywords: str, years: str = None, pnum: int = 10) -> str:
    """Fetch papers from PubMed and store them locally. Use BEFORE search_literature_tool."""
    return pubmed_search_and_store(keywords, years, pnum)


@tool
def search_literature_tool(query: str, num_results: int = 10) -> str:
    """Search stored papers with KG context. Only cite PMIDs from this output."""
    results, kg_context, has_relevant = kg_enhanced_search(query, num_results)
    
    # ====================================================
    # ANTI-HALLUCINATION GUARD
    # ====================================================
    if results == "NO_PAPERS_FOUND" or not has_relevant:
        no_data_msg = (
            "⚠️ NO RELEVANT PAPERS IN DATABASE for this query.\n"
            "DO NOT cite any PMIDs. DO NOT invent references.\n"
            "→ Tell the user: no stored papers match this topic.\n"
            "→ Suggest: use pubmed_search_and_store_tool to fetch papers first."
        )
        if kg_context:
            return f"{kg_context}\n\n{no_data_msg}"
        return no_data_msg
    
    # Normal case: relevant papers found
    output = ""
    if kg_context:
        output = f"{kg_context}\n\n"
    output += results
    output += "\n\n✅ You may cite PMIDs listed above. Do NOT cite any other PMIDs."
    return output


@tool
def get_database_stats_tool() -> str:
    """Show vector DB and knowledge graph statistics."""
    output = f"Vector DB: {vectorstore._collection.count()} chunks\n"
    if kg_manager:
        stats = kg_manager.get_stats()
        output += f"KG: {stats.get('entities', 0):,} entities, {stats.get('relations', 0):,} relations"
    return output


@tool
def verify_facts_tool(text: str) -> str:
    """Check claims in text against the knowledge graph."""
    if not kg_manager:
        return "KG not enabled"
    result = kg_manager.verify_response(text)
    return f"Claims: {result['total_claims']}, Verified: {result['verified_claims']}, " \
           f"Rate: {result['verification_rate']:.0%}"


@tool
def explore_kg_entity_tool(entity_name: str) -> str:
    """Look up an entity and its neighbors in the knowledge graph."""
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
    """Get gene function, aliases, diseases from NCBI+UniProt. Use FIRST for any gene question."""
    gene_symbol = gene_symbol.strip().upper()
    
    output = [f"**Gene Information: {gene_symbol}**\n"]
    sources_status = {"NCBI Gene": "❌ NOT QUERIED", "UniProt": "❌ NOT QUERIED"}
    found_info = False
    
    # 1. NCBI Gene Database
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
    
    # 2. UniProt
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
    
    # DATA SOURCE STATUS SUMMARY
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
# GENE COORDINATES TOOL
# =========================================

@tool
def get_gene_coordinates_tool(gene_symbol: str) -> str:
    """Get genomic coordinates (GRCh38) for a gene from Ensembl."""
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
    
    output.append("\n" + "="*50)
    output.append("**DATA SOURCE STATUS:**")
    for source, status in sources_status.items():
        output.append(f"  • {source}: {status}")
    
    return "\n".join(output)


def get_promoter_region(gene_symbol: str, upstream: int = 1500, downstream: int = 500) -> str:
    """Get promoter coordinates for a gene."""
    gene_symbol = gene_symbol.strip().upper()
    
    try:
        url = f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene_symbol}"
        headers = {"Content-Type": "application/json"}
        r = requests.get(url, headers=headers, timeout=15)
        
        if r.ok:
            data = r.json()
            chrom = data.get("seq_region_name")
            start = data.get("start")
            end = data.get("end")
            strand = data.get("strand", 1)
            
            if strand == 1:
                tss = start
                promoter_start = tss - upstream
                promoter_end = tss + downstream
            else:
                tss = end
                promoter_start = tss - downstream
                promoter_end = tss + upstream
            
            return {
                "gene": gene_symbol,
                "tss": tss,
                "strand": "+" if strand == 1 else "-",
                "promoter_location": f"chr{chrom}:{promoter_start}-{promoter_end}",
                "promoter_size": upstream + downstream,
            }
        else:
            return {"error": f"Gene not found: {gene_symbol}"}
            
    except Exception as e:
        return {"error": str(e)}


@tool
def get_promoter_coordinates_tool(gene_symbol: str, upstream: int = 1500, downstream: int = 500) -> str:
    """Get promoter region coordinates for AlphaGenome predictions."""
    result = get_promoter_region(gene_symbol, upstream, downstream)
    
    if "error" in result:
        return f"Error: {result['error']}"
    
    return (
        f"**Promoter Region for {result['gene']}**\n"
        f"  TSS position: {result['tss']:,}\n"
        f"  Strand: {result['strand']}\n"
        f"  Promoter: {result['promoter_location']}\n"
        f"  Size: {result['promoter_size']} bp\n\n"
        f"Use this location for AlphaGenome: {result['promoter_location']}"
    )


class GenomicPredictionInput(BaseModel):
    coordinates: str = Field(
        description="Genomic coordinates in 'chr:start-end' format (e.g., 'chr17:7676000-7677000')."
    )
    tissue: str = Field(
        description="Target tissue or cell type name (e.g., 'lung', 'liver', 'T-cell')."
    )
    assays: List[str] = Field(
        default=["atac"],
        description="List of assays to predict. Valid options: ['atac', 'dnase', 'rna', 'cage', 'chip_histone', 'chip_tf']."
    )

PREDICTOR = AlphaGenomePredictor(skip_validation=False)

@tool(args_schema=GenomicPredictionInput)
def predict_genomics(coordinates: str, tissue: str, assays: List[str] = ["atac"]) -> str:
    """Predict genomic signals (ATAC/RNA/ChIP) for coordinates+tissue via AlphaGenome."""
    try:
        result = PREDICTOR.predict_coordinates(
            coordinates=coordinates,
            tissue=tissue,
            assays=assays,
            sequence_length="auto"
        )
        
        from alphagenome_tool import format_results
        return format_results(result)
        
    except Exception as e:
        return f"Error running AlphaGenome prediction: {str(e)}"


# ============================================
# AGENT SETUP
# ============================================

memory = MemorySaver()

# ===========================================================================
# SYSTEM PROMPT — Optimized for 8B models
# ===========================================================================
# Design principles:
#   1. Keyword routing table instead of prose (fewer tokens, less ambiguity)
#   2. Explicit "DO NOT" rules at the top (8B models respect negatives better
#      when they come first)
#   3. No nested markdown structure (confuses small models)
#   4. Tool names are exact strings the model must emit
# ===========================================================================

system_prompt = """You are a biomedical research assistant. Use tools to answer. Never guess.

STRICT RULES:
- Never invent PMIDs. Only cite PMIDs that appear in tool output.
- If search_literature_tool says "NO RELEVANT PAPERS", tell the user and suggest fetching papers with pubmed_search_and_store_tool. Do NOT make up citations.
- Always use gene_info_tool FIRST before discussing any gene.
- AlphaGenome results are AI predictions, not experiments. Say so.

TOOL ROUTING (pick by keyword):
  "What is [GENE]?" / gene function / aliases → gene_info_tool
  "Where is [GENE]?" / coordinates / location → get_gene_coordinates_tool
  promoter / TSS / regulatory region → get_promoter_coordinates_tool
  ATAC / ChIP / accessibility / chromatin / predict signal → predict_genomics
  "Find papers" / fetch / download / PubMed search → pubmed_search_and_store_tool
  "What do papers say?" / summarize / literature → search_literature_tool
  verify / fact-check → verify_facts_tool
  explore entity / KG / knowledge graph → explore_kg_entity_tool
  stats / database size → get_database_stats_tool
  general knowledge / "what do you know" / "tell me about" → search_literature_tool

WORKFLOW FOR LITERATURE QUESTIONS:
1. Call search_literature_tool first (checks stored papers + KG).
2. If it says NO RELEVANT PAPERS → tell user, suggest pubmed_search_and_store_tool.
3. After user fetches papers → call search_literature_tool again to answer.

WORKFLOW FOR GENOMIC PREDICTIONS:
1. Get coordinates: use get_gene_coordinates_tool or get_promoter_coordinates_tool.
2. Call predict_genomics with chr:start-end, tissue name, and assay list.
3. For tissue comparisons: make TWO separate predict_genomics calls.

CITATION FORMAT: [PMID: 12345678] — only from tool output."""

tools = [
    pubmed_search_and_store_tool,
    search_literature_tool,           # ← merged check_rag + search_rag
    get_database_stats_tool,
    verify_facts_tool,
    explore_kg_entity_tool,
    get_gene_coordinates_tool,
    get_promoter_coordinates_tool,
    gene_info_tool,
    predict_genomics,
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
    print("PubMed Research Agent v3.3 - Optimized for 8B")
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
