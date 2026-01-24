import argparse
from datetime import date
from deepagents import create_deep_agent
from langchain.agents import create_agent
from langgraph_supervisor import create_supervisor
from dotenv import load_dotenv
from langchain_ollama import ChatOllama 
import os
from Bio import Entrez
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
import re
import requests
import xml.etree.ElementTree as ET
# RAG Components
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.tools import tool
import json
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import PromptTemplate
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import torch
# New imports for enhancements
import asyncio
import aiohttp
from collections import defaultdict
from rank_bm25 import BM25Okapi
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

Entrez.email = "rafailadam46@gmail.com"
pi_llm = ChatOllama(model="ministral-3:8b")
# worker_llm = ChatOllama(model='cniongolo/biomistral')
embeddings = HuggingFaceEmbeddings(
    model_name="FremyCompany/BioLORD-2023", 
    model_kwargs={'device': 'cuda'}, 
    encode_kwargs={'device': 'cuda', 'batch_size': 32}
)

# ============================================
# RECIPROCAL RANK FUSION
# ============================================

def reciprocal_rank_fusion(
    results_lists: List[List[Document]], 
    k: int = 60,
    top_n: Optional[int] = None
) -> List[Document]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.
    
    RRF score = sum(1 / (k + rank)) across all lists where document appears.
    
    Args:
        results_lists: List of ranked document lists from different retrieval methods
        k: RRF constant (default 60, as per original paper)
        top_n: Number of results to return (None = all)
    
    Returns:
        Fused and re-ranked list of documents
    """
    scores: Dict[str, float] = defaultdict(float)
    doc_map: Dict[str, Document] = {}
    
    for results in results_lists:
        for rank, doc in enumerate(results):
            # Create unique ID from content hash + pmid
            doc_id = f"{doc.metadata.get('pmid', 'unknown')}_{hash(doc.page_content[:100])}"
            
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            
            # RRF formula: 1 / (k + rank + 1)
            # rank + 1 because ranks are 0-indexed
            scores[doc_id] += 1.0 / (k + rank + 1)
    
    # Sort by RRF score descending
    sorted_doc_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    # Build result list with scores in metadata
    fused_results = []
    for doc_id in sorted_doc_ids:
        doc = doc_map[doc_id]
        # Add RRF score to metadata for debugging/transparency
        doc.metadata['rrf_score'] = scores[doc_id]
        fused_results.append(doc)
    
    if top_n:
        return fused_results[:top_n]
    return fused_results


# ============================================
# BM25 SPARSE RETRIEVER
# ============================================

class BM25SparseRetriever:
    """
    BM25-based sparse retriever that works alongside dense retrieval.
    Rebuilds index from vectorstore on initialization and can be refreshed.
    """
    
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore
        self.documents: List[Document] = []
        self.bm25: Optional[BM25Okapi] = None
        self.corpus: List[List[str]] = []
        self._build_index()
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization - lowercase and split on non-alphanumeric."""
        return re.findall(r'\w+', text.lower())
    
    def _build_index(self):
        """Build BM25 index from all documents in vectorstore."""
        try:
            collection = self.vectorstore._collection
            count = collection.count()
            
            if count == 0:
                logger.info("BM25: Vectorstore empty, skipping index build")
                return
            
            # Get all documents from collection
            all_docs = collection.get(include=['documents', 'metadatas'])
            
            self.documents = []
            self.corpus = []
            
            for i, (content, metadata) in enumerate(zip(
                all_docs.get('documents', []), 
                all_docs.get('metadatas', [])
            )):
                if content:
                    doc = Document(page_content=content, metadata=metadata or {})
                    self.documents.append(doc)
                    self.corpus.append(self._tokenize(content))
            
            if self.corpus:
                self.bm25 = BM25Okapi(self.corpus)
                logger.info(f"BM25: Built index with {len(self.corpus)} documents")
            
        except Exception as e:
            logger.error(f"BM25: Failed to build index: {e}")
            self.bm25 = None
    
    def refresh(self):
        """Rebuild the BM25 index (call after adding new documents)."""
        self._build_index()
    
    def search(self, query: str, k: int = 10) -> List[Document]:
        """
        Search using BM25.
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            List of Documents ranked by BM25 score
        """
        if not self.bm25 or not self.documents:
            logger.warning("BM25: Index not available, returning empty results")
            return []
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include if there's some relevance
                doc = self.documents[idx]
                doc.metadata['bm25_score'] = float(scores[idx])
                results.append(doc)
        
        return results


# ============================================
# HYBRID RETRIEVER
# ============================================

class HybridRetriever:
    """
    Combines dense (embedding) and sparse (BM25) retrieval using RRF.
    """
    
    def __init__(
        self, 
        vectorstore: Chroma, 
        bm25_retriever: BM25SparseRetriever,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        dense_k: int = 20,
        sparse_k: int = 20
    ):
        self.vectorstore = vectorstore
        self.bm25_retriever = bm25_retriever
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.dense_k = dense_k
        self.sparse_k = sparse_k
    
    def search(self, query: str, k: int = 10) -> List[Document]:
        """
        Perform hybrid search combining dense and sparse retrieval.
        
        Args:
            query: Search query
            k: Number of final results to return
        
        Returns:
            List of Documents ranked by fused score
        """
        results_lists = []
        
        # Dense retrieval
        try:
            dense_results = self.vectorstore.similarity_search(query, k=self.dense_k)
            if dense_results:
                results_lists.append(dense_results)
                logger.info(f"Hybrid: Dense retrieval returned {len(dense_results)} results")
        except Exception as e:
            logger.error(f"Hybrid: Dense retrieval failed: {e}")
        
        # Sparse retrieval
        try:
            sparse_results = self.bm25_retriever.search(query, k=self.sparse_k)
            if sparse_results:
                results_lists.append(sparse_results)
                logger.info(f"Hybrid: Sparse retrieval returned {len(sparse_results)} results")
        except Exception as e:
            logger.error(f"Hybrid: Sparse retrieval failed: {e}")
        
        if not results_lists:
            return []
        
        # Fuse results using RRF
        fused = reciprocal_rank_fusion(results_lists, top_n=k)
        logger.info(f"Hybrid: RRF fusion returned {len(fused)} results")
        
        return fused


# ============================================
# ASYNC PUBMED FETCHING
# ============================================

class AsyncPubMedFetcher:
    """
    Async PubMed fetcher for faster parallel retrieval.
    Respects NCBI rate limits (3 requests/second without API key, 10 with).
    """
    
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    
    def __init__(self, email: str, api_key: Optional[str] = None, max_concurrent: int = 3):
        self.email = email
        self.api_key = api_key
        self.max_concurrent = max_concurrent
        self.semaphore: Optional[asyncio.Semaphore] = None
    
    def _get_params(self, **kwargs) -> Dict[str, str]:
        """Build request params with email and optional API key."""
        params = {"email": self.email, **kwargs}
        if self.api_key:
            params["api_key"] = self.api_key
        return params
    
    async def _rate_limited_request(
        self, 
        session: aiohttp.ClientSession, 
        url: str, 
        params: Dict[str, str]
    ) -> Optional[str]:
        """Make a rate-limited request."""
        async with self.semaphore:
            try:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        return await response.text()
                    else:
                        logger.warning(f"PubMed API returned status {response.status}")
                        return None
            except Exception as e:
                logger.error(f"Request failed: {e}")
                return None
            finally:
                # Rate limiting: wait between requests
                await asyncio.sleep(0.35)  # ~3 requests per second
    
    async def search(self, term: str, retmax: int = 10) -> List[str]:
        """
        Search PubMed and return list of PMIDs.
        
        Args:
            term: Search term
            retmax: Maximum results to return
        
        Returns:
            List of PMIDs
        """
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        url = f"{self.BASE_URL}/esearch.fcgi"
        params = self._get_params(db="pubmed", term=term, retmax=str(retmax), retmode="json")
        
        async with aiohttp.ClientSession() as session:
            response = await self._rate_limited_request(session, url, params)
            if response:
                try:
                    data = json.loads(response)
                    return data.get("esearchresult", {}).get("idlist", [])
                except json.JSONDecodeError:
                    logger.error("Failed to parse search response")
        return []
    
    async def fetch_abstracts(self, pmids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Fetch abstracts for multiple PMIDs in parallel.
        
        Args:
            pmids: List of PMIDs to fetch
        
        Returns:
            Dict mapping PMID to article data
        """
        if not pmids:
            return {}
        
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        url = f"{self.BASE_URL}/efetch.fcgi"
        
        # Fetch in batches to avoid URL length limits
        batch_size = 50
        all_results = {}
        
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(pmids), batch_size):
                batch = pmids[i:i + batch_size]
                params = self._get_params(
                    db="pubmed", 
                    id=",".join(batch), 
                    rettype="abstract", 
                    retmode="xml"
                )
                
                response = await self._rate_limited_request(session, url, params)
                if response:
                    parsed = self._parse_pubmed_xml(response)
                    all_results.update(parsed)
        
        return all_results
    
    async def fetch_full_texts(self, pmids: List[str]) -> Dict[str, Optional[str]]:
        """
        Attempt to fetch full texts from PMC for given PMIDs.
        
        Args:
            pmids: List of PMIDs
        
        Returns:
            Dict mapping PMID to full text (or None if not available)
        """
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        
        # First, get PMC IDs for each PMID
        pmc_mapping = await self._get_pmc_ids(pmids)
        
        if not pmc_mapping:
            return {pmid: None for pmid in pmids}
        
        # Fetch full texts in parallel
        async with aiohttp.ClientSession() as session:
            tasks = []
            pmid_order = []
            
            for pmid, pmc_id in pmc_mapping.items():
                if pmc_id:
                    tasks.append(self._fetch_single_full_text(session, pmc_id))
                    pmid_order.append(pmid)
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                full_texts = {}
                for pmid, result in zip(pmid_order, results):
                    if isinstance(result, Exception):
                        full_texts[pmid] = None
                    else:
                        full_texts[pmid] = result
                
                # Add None for PMIDs without PMC IDs
                for pmid in pmids:
                    if pmid not in full_texts:
                        full_texts[pmid] = None
                
                return full_texts
        
        return {pmid: None for pmid in pmids}
    
    async def _get_pmc_ids(self, pmids: List[str]) -> Dict[str, Optional[str]]:
        """Get PMC IDs for PMIDs via elink."""
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        url = f"{self.BASE_URL}/elink.fcgi"
        
        mapping = {}
        
        async with aiohttp.ClientSession() as session:
            # Process in smaller batches for elink
            batch_size = 20
            for i in range(0, len(pmids), batch_size):
                batch = pmids[i:i + batch_size]
                params = self._get_params(
                    dbfrom="pubmed",
                    db="pmc",
                    id=",".join(batch),
                    linkname="pubmed_pmc",
                    retmode="json"
                )
                
                response = await self._rate_limited_request(session, url, params)
                if response:
                    try:
                        data = json.loads(response)
                        linksets = data.get("linksets", [])
                        for linkset in linksets:
                            pmid = linkset.get("ids", [None])[0]
                            if pmid:
                                pmid = str(pmid)
                                linksetdbs = linkset.get("linksetdbs", [])
                                if linksetdbs:
                                    links = linksetdbs[0].get("links", [])
                                    if links:
                                        mapping[pmid] = str(links[0])
                                    else:
                                        mapping[pmid] = None
                                else:
                                    mapping[pmid] = None
                    except (json.JSONDecodeError, KeyError, IndexError) as e:
                        logger.error(f"Failed to parse elink response: {e}")
        
        return mapping
    
    async def _fetch_single_full_text(
        self, 
        session: aiohttp.ClientSession, 
        pmc_id: str
    ) -> Optional[str]:
        """Fetch full text for a single PMC ID."""
        url = f"{self.BASE_URL}/efetch.fcgi"
        params = self._get_params(db="pmc", id=pmc_id, rettype="full", retmode="xml")
        
        response = await self._rate_limited_request(session, url, params)
        if response:
            return self._extract_text_from_pmc_xml(response)
        return None
    
    def _parse_pubmed_xml(self, xml_text: str) -> Dict[str, Dict[str, Any]]:
        """Parse PubMed XML response into structured data."""
        results = {}
        try:
            root = ET.fromstring(xml_text)
            for article in root.findall('.//PubmedArticle'):
                try:
                    pmid_elem = article.find('.//PMID')
                    if pmid_elem is None:
                        continue
                    pmid = pmid_elem.text
                    
                    title_elem = article.find('.//ArticleTitle')
                    title = title_elem.text if title_elem is not None else "Unknown"
                    
                    # Extract abstract
                    abstract_parts = []
                    for abstract_text in article.findall('.//AbstractText'):
                        if abstract_text.text:
                            abstract_parts.append(abstract_text.text)
                    abstract = " ".join(abstract_parts) if abstract_parts else "No abstract available."
                    
                    # Extract authors
                    authors = []
                    for author in article.findall('.//Author'):
                        lastname = author.find('LastName')
                        forename = author.find('ForeName')
                        if lastname is not None:
                            name = lastname.text
                            if forename is not None:
                                name += f" {forename.text}"
                            authors.append(name)
                    
                    # Extract year
                    year = None
                    pub_date = article.find('.//PubDate/Year')
                    if pub_date is not None:
                        year = pub_date.text
                    else:
                        medline_date = article.find('.//PubDate/MedlineDate')
                        if medline_date is not None and medline_date.text:
                            year_match = re.search(r'\d{4}', medline_date.text)
                            if year_match:
                                year = year_match.group()
                    
                    results[pmid] = {
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "authors": "; ".join(authors) if authors else "No authors listed",
                        "year": year
                    }
                    
                except Exception as e:
                    logger.error(f"Failed to parse article: {e}")
                    continue
                    
        except ET.ParseError as e:
            logger.error(f"XML parse error: {e}")
        
        return results
    
    def _extract_text_from_pmc_xml(self, xml_text: str) -> Optional[str]:
        """Extract body text from PMC XML."""
        try:
            root = ET.fromstring(xml_text)
            text_parts = []
            
            # Try to find body paragraphs
            for p in root.findall('.//body//p'):
                if p.text:
                    text_parts.append(p.text)
                # Also get tail text and nested text
                for child in p.iter():
                    if child.text:
                        text_parts.append(child.text)
                    if child.tail:
                        text_parts.append(child.tail)
            
            if not text_parts:
                # Fallback: get all text from body
                body = root.find('.//body')
                if body is not None:
                    text_parts = [t.strip() for t in body.itertext() if t.strip()]
            
            if text_parts:
                full_text = ' '.join(text_parts)
                full_text = re.sub(r'\s+', ' ', full_text).strip()
                return full_text
                
        except ET.ParseError as e:
            logger.error(f"PMC XML parse error: {e}")
        
        return None


# ============================================
# ENHANCED RAG COMPONENTS
# ============================================

class QueryRewriter:
    """Rewrites queries to improve retrieval accuracy"""
    
    def __init__(self, llm=None):
        self.llm = llm or pi_llm
        self.rewrite_prompt = PromptTemplate(
            input_variables=["query"],
            template="""Given the following query, generate 3 alternative search queries that would help find relevant biomedical literature. 
            Focus on expanding abbreviations, adding synonyms, and including related medical terms.
            
            Original query: {query}
            
            Generate 3 alternative queries (one per line):"""
        )
    
    def rewrite(self, query: str) -> List[str]:
        """Generate multiple query variations"""
        try:
            response = self.llm.invoke(self.rewrite_prompt.format(query=query))
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Parse the response to get individual queries
            lines = response_text.strip().split('\n')
            queries = [query]  # Include original
            for line in lines:
                # Clean up the line
                clean_line = re.sub(r'^[\d\.\-\*]+\s*', '', line.strip())
                if clean_line and len(clean_line) > 5:
                    queries.append(clean_line)
            
            return queries[:4]  # Return max 4 queries
        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}")
            return [query]  # Fallback to original


# ============================================
# RAG SETUP
# ============================================

os.makedirs("./pubmed_rag_db", exist_ok=True)

vectorstore = Chroma(
    persist_directory="./pubmed_rag_db",
    embedding_function=embeddings,
    collection_name="pubmed_papers",
    collection_metadata={"hnsw:space": "cosine"}
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Initialize components
query_rewriter = QueryRewriter(pi_llm)
enc_model = HuggingFaceCrossEncoder(
    model_name='ncbi/MedCPT-Cross-Encoder', 
    model_kwargs={'device': 'cuda'}
)
compressor = CrossEncoderReranker(model=enc_model, top_n=10)

# Initialize BM25 and Hybrid retriever
bm25_retriever = BM25SparseRetriever(vectorstore)
hybrid_retriever = HybridRetriever(
    vectorstore=vectorstore,
    bm25_retriever=bm25_retriever,
    dense_weight=0.6,
    sparse_weight=0.4,
    dense_k=20,
    sparse_k=20
)

# Async PubMed fetcher
async_fetcher = AsyncPubMedFetcher(
    email=Entrez.email,
    api_key=os.getenv("NCBI_API_KEY"),  # Optional: set in .env for higher rate limits
    max_concurrent=3
)


# ============================================
# HELPER FUNCTIONS
# ============================================

def extract_authors(article):
    """Extract authors from a PubMed article."""
    authors = []
    try:
        if 'AuthorList' in article['MedlineCitation']['Article']:
            author_list = article['MedlineCitation']['Article']['AuthorList']
            for author in author_list:
                if 'CollectiveName' in author:
                    authors.append(author['CollectiveName'])
                else:
                    name_parts = []
                    if 'LastName' in author:
                        name_parts.append(author['LastName'])
                    if 'ForeName' in author:
                        name_parts.append(author['ForeName'])
                    elif 'Initials' in author:
                        name_parts.append(author['Initials'])
                    if name_parts:
                        authors.append(' '.join(name_parts))
    except (KeyError, TypeError):
        pass
    return '; '.join(authors) if authors else "No authors listed"


async def async_pubmed_search_and_store(
    keywords: str, 
    years: str = None, 
    pnum: int = 5
) -> str:
    """
    Async version of PubMed search and store with parallel fetching.
    
    Args:
        keywords: Search keywords
        years: Single year, comma-separated, or range (e.g., "2022-2024")
        pnum: Number of papers per year
    
    Returns:
        Summary of papers found and stored
    """
    # Parse years
    year_list = []
    if not years:
        year_list = [str(date.today().year)]
    elif "-" in years:
        start_year, end_year = years.split("-")
        year_list = [str(y) for y in range(int(start_year.strip()), int(end_year.strip()) + 1)]
    elif "," in years:
        year_list = [y.strip() for y in years.split(",")]
    else:
        year_list = [years.strip()]
    
    all_results = []
    total_stored = 0
    total_full_text = 0
    all_papers = []
    
    for year in year_list:
        logger.info(f"Searching PubMed for '{keywords}' in year {year}...")
        search_term = f"({keywords}) AND {year}[pdat]"
        
        # Async search
        pmids = await async_fetcher.search(search_term, retmax=pnum)
        
        if not pmids:
            all_results.append(f"Year {year}: No papers found")
            continue
        
        logger.info(f"Found {len(pmids)} PMIDs, fetching abstracts...")
        
        # Async fetch abstracts
        articles = await async_fetcher.fetch_abstracts(pmids)
        
        if not articles:
            all_results.append(f"Year {year}: Failed to fetch papers")
            continue
        
        # Async fetch full texts
        logger.info("Attempting to fetch full texts...")
        full_texts = await async_fetcher.fetch_full_texts(pmids)
        
        year_papers = []
        year_full_text = 0
        
        for pmid, article in articles.items():
            try:
                title = article['title']
                authors = article['authors']
                abstract = article['abstract']
                full_text = full_texts.get(pmid)
                
                if full_text:
                    year_full_text += 1
                    logger.info(f"  Full text retrieved for PMID:{pmid}")
                
                # Create content for vector store
                content = f"PMID: {pmid}\nTitle: {title}\n"
                content += f"Authors: {authors}\nYear: {year}\n"
                if full_text:
                    content += f"\nFull Text: {full_text}"
                else:
                    content += f"\nAbstract: {abstract}"
                
                # Split and store
                chunks = text_splitter.split_text(content)
                documents = [
                    Document(
                        page_content=chunk,
                        metadata={
                            "pmid": pmid,
                            "title": title,
                            "authors": authors,
                            "year": year,
                            "chunk_index": i,
                            "source": "pubmed",
                            "has_full_text": bool(full_text)
                        }
                    )
                    for i, chunk in enumerate(chunks)
                ]
                
                vectorstore.add_documents(documents)
                year_papers.append(f"{title}-(PMID:{pmid})")
                all_papers.append(f"{title}-(PMID:{pmid})\n")
                
            except Exception as e:
                logger.error(f"Error processing article {pmid}: {e}")
                continue
        
        if year_papers:
            all_results.append(
                f"Year {year}: Found {len(year_papers)} papers, "
                f"{year_full_text} with full text"
            )
            total_stored += len(year_papers)
            total_full_text += year_full_text
    
    # Refresh BM25 index after adding new documents
    bm25_retriever.refresh()
    logger.info("BM25 index refreshed")
    
    # Final summary
    summary = f"SEARCH COMPLETE:\n"
    summary += f"Total papers stored: {total_stored}\n"
    summary += f"Total with full text: {total_full_text}\n\n"
    summary += f"Titles with PMIDs: {all_papers}\n\n"
    
    return summary


def pubmed_search_and_store_multi_year(keywords: str, years: str = None, pnum: int = 10) -> str:
    """
    Wrapper for async search - runs the async function in event loop.
    """
    try:
        # Check if we're already in an event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, create a new task
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(
                async_pubmed_search_and_store(keywords, years, pnum)
            )
        else:
            return asyncio.run(async_pubmed_search_and_store(keywords, years, pnum))
    except RuntimeError:
        # No event loop, create one
        return asyncio.run(async_pubmed_search_and_store(keywords, years, pnum))


# Create tools
pubmed_search_and_store_tool = tool(
    pubmed_search_and_store_multi_year,
    description="Search PubMed for papers and store them. Supports single year, multiple years (comma-separated), or year ranges (e.g., 2022-2024). Uses async fetching for faster results."
)


def enhanced_search_rag_database(
    query: str, 
    num_results: int = 10,
    use_reranking: bool = True,
    use_hybrid: bool = True,
    use_query_rewrite: bool = True
) -> str:
    """
    Enhanced search with hybrid retrieval, RRF fusion, and reranking.
    
    Args:
        query: Search query or PMID
        num_results: Number of results to return
        use_reranking: Whether to use cross-encoder reranking
        use_hybrid: Whether to use hybrid (dense + sparse) retrieval
        use_query_rewrite: Whether to rewrite query for better retrieval
    
    Returns:
        Formatted search results
    """
    logger.info(f"Enhanced search for: '{query}'")
    
    # Clean up query
    clean_query = query.replace("PMID:", "").replace("pmid:", "").strip()
    
    # Check if this is a PMID query
    is_pmid_query = all(part.strip().isdigit() for part in clean_query.split(','))
    
    if is_pmid_query:
        # PMID query - disable all enhancements for precise retrieval
        logger.info("PMID query detected - disabling enhancements")
        
        pmids = [p.strip() for p in clean_query.split(',')]
        results = []
        for pmid in pmids:
            try:
                collection = vectorstore._collection
                pmid_docs = collection.get(where={"pmid": pmid})
                
                if pmid_docs and pmid_docs['ids']:
                    for i, doc_id in enumerate(pmid_docs['ids']):
                        doc = Document(
                            page_content=pmid_docs['documents'][i],
                            metadata=pmid_docs['metadatas'][i] if pmid_docs['metadatas'] else {}
                        )
                        results.append(doc)
                    
                    # Sort by chunk index
                    pmid_results = [r for r in results if r.metadata.get('pmid') == pmid]
                    pmid_results.sort(key=lambda x: x.metadata.get('chunk_index', 0))
                    
            except Exception as e:
                logger.error(f"Error fetching PMID {pmid}: {e}")
                # Fallback to similarity search
                pmid_results = vectorstore.similarity_search(f"PMID: {pmid}", k=50)
                pmid_results = [r for r in pmid_results if r.metadata.get('pmid') == pmid]
                pmid_results.sort(key=lambda x: x.metadata.get('chunk_index', 0))
                results.extend(pmid_results)
        
        if not results:
            return f"No paper found with PMID: {clean_query}"
            
    else:
        # Keyword search - apply all enhancements
        logger.info(f"Options: rerank={use_reranking}, hybrid={use_hybrid}, rewrite={use_query_rewrite}")
        
        # Query rewriting
        if use_query_rewrite:
            queries = query_rewriter.rewrite(query)
            logger.info(f"Rewritten queries: {queries}")
        else:
            queries = [query]
        
        all_results_lists = []
        
        # Search with all query variations
        for q in queries:
            try:
                if use_hybrid:
                    # Use hybrid retriever (dense + sparse with RRF)
                    search_results = hybrid_retriever.search(q, k=num_results * 2)
                else:
                    # Dense only
                    search_results = vectorstore.similarity_search(q, k=num_results * 2)
                
                if search_results:
                    all_results_lists.append(search_results)
                    
            except Exception as e:
                logger.error(f"Search failed for query '{q}': {e}")
        
        if not all_results_lists:
            return "No relevant papers found in local database."
        
        # Fuse results from multiple queries using RRF
        if len(all_results_lists) > 1:
            results = reciprocal_rank_fusion(all_results_lists, top_n=num_results * 2)
            logger.info(f"Fused {len(all_results_lists)} query results via RRF")
        else:
            results = all_results_lists[0]
        
        # Deduplicate by content
        seen_content = set()
        unique_results = []
        for doc in results:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(doc)
        
        # Apply cross-encoder reranking
        if use_reranking and len(unique_results) > 1:
            logger.info(f"Reranking {len(unique_results)} results with cross-encoder...")
            results = compressor.compress_documents(unique_results, query)[:num_results]
        else:
            results = unique_results[:num_results]
    
    if not results:
        return "No relevant papers found in local database."
    
    # Group and format results
    papers_dict = {}
    for doc in results:
        pmid = doc.metadata.get("pmid", "unknown")
        if pmid not in papers_dict:
            papers_dict[pmid] = {
                "pmid": pmid,
                "title": doc.metadata.get("title", "Unknown"),
                "authors": doc.metadata.get("authors", "Unknown"),
                "year": doc.metadata.get("year", "Unknown"),
                "has_full_text": doc.metadata.get("has_full_text", False),
                "rrf_score": doc.metadata.get("rrf_score", 0),
                "chunks": []
            }
        papers_dict[pmid]["chunks"].append(doc.page_content)
    
    # Sort by RRF score if available
    if not is_pmid_query:
        sorted_papers = sorted(
            papers_dict.items(), 
            key=lambda x: x[1].get("rrf_score", 0), 
            reverse=True
        )
    else:
        sorted_papers = list(papers_dict.items())
    
    # Format output
    output = []
    output.append(f"Found {len(sorted_papers)} relevant paper(s)")
    
    if not is_pmid_query:
        enhancements = []
        if use_hybrid:
            enhancements.append("hybrid search")
        if use_reranking:
            enhancements.append("cross-encoder reranking")
        if use_query_rewrite:
            enhancements.append("query expansion")
        if enhancements:
            output.append(f"   (Enhanced with: {', '.join(enhancements)})")
    else:
        output.append("   (Direct PMID retrieval - no enhancements applied)")
    
    output.append("\n")
    
    display_limit = len(sorted_papers) if is_pmid_query else min(num_results, len(sorted_papers))
    
    for idx, (pmid, info) in enumerate(sorted_papers[:display_limit], 1):
        content_type = "full text" if info['has_full_text'] else "abstract"
        combined_content = "\n\n".join(info['chunks'])
        
        if is_pmid_query and len(sorted_papers) == 1:
            display_content = combined_content
        elif len(sorted_papers) > 1:
            max_length = 500
            if len(combined_content) > max_length:
                display_content = combined_content[:max_length] + "..."
            else:
                display_content = combined_content
        else:
            display_content = combined_content
        
        output.append(
            f"\n[{idx}] PMID: {pmid}\n"
            f"Title: {info['title']}\n"
            f"Authors: {info['authors']}\n"
            f"Year: {info['year']} | Type: {content_type}\n"
            f"\nContent:\n{display_content}\n"
            f"{'='*60}"
        )
    
    if not is_pmid_query and len(sorted_papers) > num_results:
        output.append(f"\nShowing top {num_results} of {len(sorted_papers)} results.")
    
    return "".join(output)


def search_rag_database(query: str, num_results: int = 10) -> str:
    """Search with smart enhancement detection."""
    clean_query = query.replace("PMID:", "").replace("pmid:", "").strip()
    is_pmid_query = all(part.strip().isdigit() for part in clean_query.split(','))
    
    if is_pmid_query:
        return enhanced_search_rag_database(
            query, num_results,
            use_reranking=False,
            use_hybrid=False,
            use_query_rewrite=False
        )
    else:
        return enhanced_search_rag_database(
            query, num_results,
            use_reranking=True,
            use_hybrid=True,
            use_query_rewrite=True
        )


search_rag_database_tool = tool(
    search_rag_database,
    description="Search RAG database with hybrid retrieval (dense + sparse), RRF fusion, and cross-encoder reranking"
)

enhanced_search_tool = tool(
    enhanced_search_rag_database,
    description="Advanced search with control over hybrid retrieval, reranking, and query rewriting"
)


def check_rag_for_topic(keywords: str) -> str:
    """Checks if papers on this topic exist in the RAG database."""
    try:
        collection = vectorstore._collection
        count = collection.count()
        
        if count == 0:
            return "Database is empty. No papers stored yet."
        
        # Use hybrid search for better topic matching
        results = hybrid_retriever.search(keywords, k=5)
        
        if not results:
            return f"No papers found for '{keywords}' in database ({count} total chunks)."
        
        papers_info = {}
        for doc in results:
            pmid = doc.metadata.get("pmid")
            if pmid and pmid not in papers_info:
                papers_info[pmid] = {
                    "title": doc.metadata.get("title"),
                    "year": doc.metadata.get("year")
                }
        
        output = f"Found {len(papers_info)} relevant papers in database:\n"
        for pmid, info in papers_info.items():
            output += f"- PMID: {pmid} | {info['title'][:60]}... ({info['year']})\n"
        
        return output
    except Exception as e:
        return f"Error checking database: {e}"


check_rag_for_topic_tool = tool(check_rag_for_topic)


def remove_from_rag_database(identifier: str, search_type: str = "auto") -> str:
    """
    Remove entries from the RAG database based on PMID or query.
    """
    logger.info(f"Attempting to remove from database: '{identifier}' (type: {search_type})")
    
    if search_type == "auto":
        clean_id = identifier.replace("PMID:", "").replace("pmid:", "").strip()
        if all(part.strip().isdigit() for part in clean_id.split(',')):
            search_type = "pmid"
        else:
            search_type = "query"
    
    try:
        collection = vectorstore._collection
        
        if search_type == "pmid":
            pmids = [p.strip() for p in identifier.replace("PMID:", "").replace("pmid:", "").strip().split(',')]
            total_deleted = 0
            
            for pmid in pmids:
                results = collection.get(where={"pmid": pmid})
                
                if results and results['ids']:
                    collection.delete(ids=results['ids'])
                    deleted_count = len(results['ids'])
                    total_deleted += deleted_count
                    logger.info(f"Deleted {deleted_count} chunks for PMID: {pmid}")
            
            # Refresh BM25 index after deletion
            bm25_retriever.refresh()
            
            if total_deleted > 0:
                return f"Successfully deleted {total_deleted} chunks from {len(pmids)} PMID(s)"
            else:
                return f"No documents found for PMIDs: {', '.join(pmids)}"
                
        elif search_type == "query":
            results = hybrid_retriever.search(identifier, k=20)
            
            if not results:
                return f"No documents found matching query: '{identifier}'"
            
            pmids_to_delete = {}
            for doc in results:
                pmid = doc.metadata.get("pmid", "unknown")
                title = doc.metadata.get("title", "Unknown")
                if pmid not in pmids_to_delete:
                    pmids_to_delete[pmid] = title
            
            output = f"Found {len(pmids_to_delete)} paper(s) matching '{identifier}':\n"
            for pmid, title in pmids_to_delete.items():
                output += f"  - PMID: {pmid} | {title[:60]}...\n"
            
            total_deleted = 0
            for pmid in pmids_to_delete.keys():
                results = collection.get(where={"pmid": pmid})
                if results and results['ids']:
                    collection.delete(ids=results['ids'])
                    total_deleted += len(results['ids'])
            
            # Refresh BM25 index after deletion
            bm25_retriever.refresh()
            
            output += f"\nDeleted {total_deleted} total chunks from {len(pmids_to_delete)} papers."
            return output
            
    except Exception as e:
        return f"Error removing from database: {e}"


remove_from_rag_tool = tool(remove_from_rag_database)


def get_database_stats() -> str:
    """Get enhanced statistics about the RAG database."""
    try:
        collection = vectorstore._collection
        count = collection.count()
        
        if count == 0:
            return "Database is empty (0 chunks, 0 papers)"
        
        sample_results = vectorstore.similarity_search("", k=min(100, count))
        
        unique_pmids = set()
        unique_titles = set()
        years = []
        has_full_text_count = 0
        
        for doc in sample_results:
            pmid = doc.metadata.get("pmid")
            title = doc.metadata.get("title")
            year = doc.metadata.get("year")
            has_full_text = doc.metadata.get("has_full_text", False)
            
            if pmid:
                unique_pmids.add(pmid)
            if title:
                unique_titles.add(title[:50])
            if year:
                years.append(int(year))
            if has_full_text:
                has_full_text_count += 1
        
        output = f"Database Statistics:\n"
        output += f"├─ Total chunks: {count}\n"
        output += f"├─ Estimated papers: {len(unique_pmids)}\n"
        output += f"├─ Papers with full text: ~{has_full_text_count}\n"
        
        if years:
            output += f"├─ Year range: {min(years)} - {max(years)}\n"
        
        output += f"├─ BM25 index: {'✓ Active' if bm25_retriever.bm25 else '✗ Not built'}\n"
        output += f"└─ Features: ✓ Hybrid Search ✓ RRF Fusion ✓ Reranking ✓ Query Rewriting\n"
        
        if unique_titles:
            output += f"\nSample papers:\n"
            for title in list(unique_titles)[:3]:
                output += f"  • {title}...\n"
        
        return output
    except Exception as e:
        return f"Error getting stats: {e}"


get_database_stats_tool = tool(get_database_stats)


# ============================================
# AGENT SETUP
# ============================================

memory = MemorySaver()

system_prompt = '''You are a PubMed research assistant. You help users find and analyze scientific papers.

CRITICAL RULE - READ THIS FIRST:
You can ONLY cite papers that appear in tool results. If a PMID is not in the tool output, you do not know about it.
When uncertain, say "I don't have information about that" rather than guessing.

TOOLS YOU HAVE:
1. pubmed_search_and_store_tool - Search PubMed and add papers to database
2. search_rag_database_tool - Find papers in your database
3. check_rag_for_topic_tool - Check if a topic exists in database
4. remove_from_rag_tool - Delete papers from database
5. get_database_stats_tool - Show database statistics

WORKFLOW:
1. User asks about a topic
2. Use check_rag_for_topic_tool to see if you have papers
3. If yes → use search_rag_database_tool to retrieve them
4. If no → use pubmed_search_and_store_tool to fetch new papers
5. Answer using ONLY the retrieved content

RESPONSE FORMAT:
When citing a paper, use this format:
  [PMID:NUMBER] "Exact Title From Tool Output" - First Author et al.

Example format ONLY (do not use these fake PMIDs):
  [PMID:00000001] "Example Paper Title" - Smith et al.
  
Only use PMIDs that appear in your tool results. Never invent a PMID.

FORBIDDEN:
- Do not cite any PMID not in tool output
- Do not copy topics from this prompt into searches
- Do not reference papers you have not retrieved

REMEMBER: No PMID in tool output = You don't know about it. Never guess.'''

tools = [
    pubmed_search_and_store_tool,
    search_rag_database_tool,
    enhanced_search_tool,
    check_rag_for_topic_tool,
    remove_from_rag_tool,
    get_database_stats_tool
]

pubmed_agent = create_agent(
    model=pi_llm,
    tools=tools,
    system_prompt=system_prompt,
    checkpointer=memory,
)


def main():
    print("PubMed Research Assistant (Enhanced)")
    print("Features: Hybrid Search | RRF Fusion | Async Fetching | Cross-Encoder Reranking")
    print("Commands: 'exit' to quit, 'stats' for database info, 'clear' to clear screen\n")
    
    stats_result = get_database_stats()
    print(f"{stats_result}\n")
    
    config = {"configurable": {"thread_id": "cli-thread"}, "recursion_limit": 250}
    
    while True:
        try:
            user_input = input(">>> ").strip()
            
            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break
            
            if user_input.lower() == "stats":
                print(f"\n{get_database_stats()}\n")
                continue
            
            if user_input.lower() == "clear":
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            
            if user_input.lower() == "refresh":
                print("Refreshing BM25 index...")
                bm25_retriever.refresh()
                print("Done!\n")
                continue
            
            # Test command for direct multi-year search
            if user_input.lower().startswith("test multi"):
                parts = user_input[10:].strip().split()
                keywords = parts[0] if parts else "CRISPR"
                years = parts[1] if len(parts) > 1 else "2022-2024"
                print(f"\nTesting async multi-year search: '{keywords}' for years {years}...")
                result = pubmed_search_and_store_multi_year(keywords, years, pnum=3)
                print(result)
                print(f"\n{get_database_stats()}\n")
                continue
            
            # Regular agent interaction
            print("\nProcessing...")
            
            result = pubmed_agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config={"configurable": {"thread_id": "cli-thread"}, "recursion_limit": 500}
            )
            
            for msg in reversed(result['messages']):
                if isinstance(msg, AIMessage) and msg.content:
                    print(f"\nAI: {msg.content}\n")
                    break
                else:
                    print(f"Agent ended with: {msg.pretty_print()}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
