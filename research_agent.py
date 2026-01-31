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

# Import knowledge graph module
from knowledge_graph import KnowledgeGraphManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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


# ============================================
# AGENT SETUP
# ============================================

memory = MemorySaver()

system_prompt = """You are a PubMed research assistant with Knowledge Graph integration.

TOOLS:
1. pubmed_search_and_store_tool - Search PubMed and store papers
2. search_rag_database_tool - Search stored papers with KG enhancement  
3. check_rag_for_topic_tool - Check if papers exist for a topic
4. get_database_stats_tool - Get database statistics
5. verify_facts_tool - Verify claims against knowledge graph
6. explore_kg_entity_tool - Explore entities in knowledge graph

RULES:
- Only cite PMIDs from tool results
- Use KG context when provided
- Never invent facts or citations"""

tools = [
    pubmed_search_and_store_tool,
    search_rag_database_tool,
    check_rag_for_topic_tool,
    get_database_stats_tool,
    verify_facts_tool,
    explore_kg_entity_tool,
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

