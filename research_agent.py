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
from langchain_tavily import TavilySearch
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

load_dotenv()

Entrez.email = "rafailadam46@gmail.com"
pi_llm = ChatOllama(model="qwen3:8b")
worker_llm = ChatOllama(model='qwen3:1.7b')
embeddings = HuggingFaceEmbeddings(model_name="FremyCompany/BioLORD-2023")

# ============================================
# ENHANCED RAG COMPONENTS
# ============================================
# Query rewriter for better retrieval
class QueryRewriter:
    """Rewrites queries to improve retrieval accuracy"""
    
    def __init__(self, llm=None):
        self.llm = llm or worker_llm
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
            print(f"Query rewrite failed: {e}")
            return [query]  # Fallback to original

# ============================================
# RAG SETUP - FIXED PERSISTENCE AND RETRIEVER
# ============================================

os.makedirs("./pubmed_rag_db", exist_ok=True)

vectorstore = Chroma(
    persist_directory="./pubmed_rag_db",
    embedding_function=embeddings,
    collection_name="pubmed_papers",
    collection_metadata={"hnsw:space": "cosine"}
)

# CREATE BASE RETRIEVER 
base_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 20}  # Retrieve more docs initially for reranking
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Initialize enhanced components with proper order
query_rewriter = QueryRewriter(worker_llm)
enc_model = HuggingFaceCrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-6-v2')
compressor = CrossEncoderReranker(model=enc_model, top_n=5)

# Now create compression retriever with the base_retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=base_retriever  # Now this is defined!
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


def pubmed_search_and_store_multi_year(keywords: str, years: str = None, pnum: int = 10) -> str:
    """
    Enhanced version that handles multiple years.
    
    Args:
        keywords: Search keywords
        years: Can be a single year ("2024"), multiple years ("2022,2023,2024"), 
               or a range ("2022-2024")
        pnum: Number of papers per year
    
    Returns:
        Summary of all papers found and stored
    """
    # Parse years input
    year_list = []
    
    if not years:
        year_list = [str(date.today().year)]
    elif "-" in years:
        # Handle range like "2022-2024"
        start_year, end_year = years.split("-")
        year_list = [str(y) for y in range(int(start_year.strip()), int(end_year.strip()) + 1)]
    elif "," in years:
        # Handle comma-separated like "2022,2023,2024"
        year_list = [y.strip() for y in years.split(",")]
    else:
        # Single year
        year_list = [years.strip()]
    
    all_results = []
    total_stored = 0
    total_full_text = 0
    all_papers = []
    
    for year in year_list:
        print(f"\nSearching PubMed for '{keywords}' in year {year}...")
        # Build search term
        search_term = f"({keywords}) AND {year}[pdat]"
        
        # Search PubMed
        try:
            handle = Entrez.esearch(db="pubmed", term=search_term, retmax=pnum, sort="relevance")
            record = Entrez.read(handle)
            handle.close()
        except Exception as e:
            all_results.append(f"Year {year}: Search failed - {e}")
            continue
        
        id_list = record["IdList"]
        if not id_list:
            all_results.append(f"Year {year}: No papers found")
            continue
        
        # Fetch and process papers
        try:
            handle = Entrez.efetch(db="pubmed", id=id_list, rettype="abstract", retmode="xml")
            articles = Entrez.read(handle)["PubmedArticle"]
            handle.close()
        except Exception as e:
            all_results.append(f"Year {year}: Fetch failed - {e}")
            continue
        
        year_papers = []
        year_full_text = 0
        
        for article in articles:
            try:
                pmid = str(article['MedlineCitation']['PMID'])
                title = article['MedlineCitation']['Article']['ArticleTitle']
                authors = extract_authors(article)
                
                # Extract abstract
                abstract = "No abstract available."
                if 'Abstract' in article['MedlineCitation']['Article']:
                    abstract_list = article['MedlineCitation']['Article']['Abstract']['AbstractText']
                    abstract = " ".join(str(a) for a in abstract_list)
                
                # Try to fetch full text
                full_text = None
                try:
                    handle = Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid, linkname="pubmed_pmc")
                    record = Entrez.read(handle)
                    handle.close()
                    
                    if record and record[0].get("LinkSetDb"):
                        pmc_id_list = [link['Id'] for link in record[0]['LinkSetDb'][0]['Link']]
                        if pmc_id_list:
                            pmc_id = pmc_id_list[0]
                            handle = Entrez.efetch(db="pmc", id=pmc_id, rettype="full", retmode="xml")
                            xml_data = handle.read()
                            handle.close()
                            
                            if xml_data:
                                root = ET.fromstring(xml_data)
                                text_parts = [p.text for p in root.findall('.//body//p') if p.text]
                                if not text_parts:
                                    body = root.find('.//body')
                                    if body is not None:
                                        text_parts = [text for text in body.itertext() if text.strip()]
                                if text_parts:
                                    full_text = ' '.join(text_parts)
                                    full_text = re.sub(r'\s+', ' ', full_text).strip()
                                    year_full_text += 1
                                    print(f"  Full text retrieved for PMID:{pmid}")
                except Exception:
                    pass
                
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
                print(f"  Error processing article: {e}")
                continue
        
        # Summary for this year
        if year_papers:
            all_results.append(
                f"Year {year}: Found {len(year_papers)} papers, "
                f"{year_full_text} with full text"
            )
            total_stored += len(year_papers)
            total_full_text += year_full_text
    
    # Final summary
    summary = f"SEARCH COMPLETE:\n"
    summary += f"Total papers stored: {total_stored}\n"
    summary += f"Total with full text: {total_full_text}\n\n"
    summary += f"Titles with PMIDS: {all_papers}\n\n"
    
    return summary


# Original function kept for backward compatibility
def pubmed_search_and_store(keywords: str, year: str = None, pnum: int = 10) -> str:
    """Original single-year search function."""
    return pubmed_search_and_store_multi_year(keywords, year, pnum)


# Create tools
pubmed_search_and_store_tool = tool(
    pubmed_search_and_store_multi_year,
    description="Search PubMed for papers and store them. Supports single year, multiple years (comma-separated), or year ranges (e.g., 2022-2024)"
)

def enhanced_search_rag_database(
    query: str, 
    num_results: int = 10,
    use_reranking: bool = True,
    use_compression: bool = True,
    use_query_rewrite: bool = True
) -> str:
    """
    Enhanced search with reranking and compression capabilities.
    
    Args:
        query: Search query or PMID
        num_results: Number of results to return
        use_reranking: Whether to use cross-encoder reranking
        use_compression: Whether to compress results
        use_query_rewrite: Whether to rewrite query for better retrieval
    
    Returns:
        Formatted search results
    """
    print(f"Enhanced search for: '{query}'")
    
    # Clean up query
    clean_query = query.replace("PMID:", "").replace("pmid:", "").strip()
    
    # Check if this is a PMID query
    is_pmid_query = all(part.strip().isdigit() for part in clean_query.split(','))
    
    if is_pmid_query:
        # CRITICAL FIX: For PMID queries, disable all enhancements
        print(f"   PMID query detected - disabling enhancements for precise retrieval")
        
        pmids = [p.strip() for p in clean_query.split(',')]
        results = []
        for pmid in pmids:
            try:
                # Use a filter-based approach to get ONLY documents with this PMID
                # First, get the collection directly for more precise filtering
                collection = vectorstore._collection
                
                # Get ALL chunks for this specific PMID using metadata filter
                pmid_docs = collection.get(
                    where={"pmid": pmid}
                )
                
                if pmid_docs and pmid_docs['ids']:
                    # Convert back to Document objects
                    for i, doc_id in enumerate(pmid_docs['ids']):
                        doc = Document(
                            page_content=pmid_docs['documents'][i] if 'documents' in pmid_docs else pmid_docs['documents'][i],
                            metadata=pmid_docs['metadatas'][i] if 'metadatas' in pmid_docs else {}
                        )
                        results.append(doc)
                
                # Sort by chunk index to maintain order
                pmid_results = [r for r in results if r.metadata.get('pmid') == pmid]
                pmid_results.sort(key=lambda x: x.metadata.get('chunk_index', 0))
                
            except Exception as e:
                print(f"Error fetching PMID {pmid}: {e}")
                # Fallback to similarity search if direct collection access fails
                pmid_results = vectorstore.similarity_search(
                    f"PMID: {pmid}", 
                    k=50  # Get many to ensure we get all chunks
                )
                pmid_results = [r for r in pmid_results if r.metadata.get('pmid') == pmid]
                pmid_results.sort(key=lambda x: x.metadata.get('chunk_index', 0))
                results.extend(pmid_results)
        
        if not results:
            return f"No paper found with PMID: {clean_query}"
        
        # For PMID queries, DO NOT apply any reranking or compression
        # Just return all chunks for the requested paper(s)
            
    else:
        # Enhanced keyword search - apply all enhancements
        print(f"   Options: rerank={use_reranking}, compress={use_compression}, rewrite={use_query_rewrite}")
        
        all_results = []
        
        # Query rewriting
        if use_query_rewrite:
            queries = query_rewriter.rewrite(query)
            print(f"   Rewritten queries: {queries}")
        else:
            queries = [query]
        
        # Search with all query variations
        for q in queries:
            try:
                # Increase initial retrieval for better reranking
                k_retrieve = num_results * 3 if use_reranking else num_results
                search_results = vectorstore.similarity_search_with_score(q, k=k_retrieve)
                all_results.extend(search_results)
            except Exception as e:
                print(f"Search failed for query '{q}': {e}")
        
        # Deduplicate by content
        seen_content = set()
        unique_results = []
        for doc, score in all_results:
            content_hash = hash(doc.page_content[:100])  # Use first 100 chars as hash
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(doc)
        
        # Apply reranking and/or compression ONLY for keyword searches
        if use_reranking and len(unique_results) > 1:
            print(f"   Reranking {len(unique_results)} results...")
            if use_compression:
                # Use the compression_retriever which combines both
                results = compression_retriever.invoke(query)
            else:
                # Just rerank without compression
                results = compressor.compress_documents(unique_results, query)
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
                "relevance_score": doc.metadata.get("relevance_score", 0),
                "chunks": []
            }
        papers_dict[pmid]["chunks"].append(doc.page_content)
    
    # Sort by relevance score if available (but not for PMID queries)
    if not is_pmid_query:
        sorted_papers = sorted(
            papers_dict.items(), 
            key=lambda x: x[1].get("relevance_score", 0), 
            reverse=True
        )
    else:
        # For PMID queries, maintain the order requested
        sorted_papers = list(papers_dict.items())
    
    # Format output
    output = []
    output.append(f"Found {len(sorted_papers)} relevant paper(s)")
    
    # Only show enhancement info for keyword searches
    if not is_pmid_query and (use_reranking or use_compression):
        enhancements = []
        if use_reranking:
            enhancements.append("reranking")
        if use_compression:
            enhancements.append("compression")
        output.append(f"   (Enhanced with: {', '.join(enhancements)})")
    elif is_pmid_query:
        output.append(f"   (Direct PMID retrieval - no enhancements applied)")
    
    output.append("\n")
    
    # For PMID queries, show ALL content; for keyword searches, limit based on num_results
    display_limit = len(sorted_papers) if is_pmid_query else min(num_results, len(sorted_papers))
    
    for idx, (pmid, info) in enumerate(sorted_papers[:display_limit], 1):
        content_type = "full text" if info['has_full_text'] else "abstract"
        
        # Combine chunks
        combined_content = "\n\n".join(info['chunks'])
        
        # For PMID queries with single paper, show full content
        # For multiple papers or keyword searches, limit display length
        if is_pmid_query and len(sorted_papers) == 1:
            # Show full content for single PMID query
            display_content = combined_content
        elif len(sorted_papers) > 1:
            # Limit for multiple papers
            max_length = 500
            if len(combined_content) > max_length:
                display_content = combined_content[:max_length] + "..."
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


# Updated wrapper function to be smarter about when to apply enhancements
def search_rag_database(query: str, num_results: int = 10) -> str:
    """
    Original search function with smart enhancement detection.
    Automatically disables enhancements for PMID queries.
    """
    # Check if this is a PMID query
    clean_query = query.replace("PMID:", "").replace("pmid:", "").strip()
    is_pmid_query = all(part.strip().isdigit() for part in clean_query.split(','))
    
    # Disable enhancements for PMID queries
    if is_pmid_query:
        return enhanced_search_rag_database(
            query, 
            num_results,
            use_reranking=False,  # Disabled for PMID
            use_compression=False,  # Disabled for PMID
            use_query_rewrite=False  # Disabled for PMID
        )
    else:
        # Enable all enhancements for keyword searches
        return enhanced_search_rag_database(
            query, 
            num_results,
            use_reranking=True,
            use_compression=True,
            use_query_rewrite=True
        )

# Wrapper for backward compatibility
def search_rag_database(query: str, num_results: int = 10) -> str:
    """Original search function with enhanced capabilities enabled by default"""
    return enhanced_search_rag_database(
        query, 
        num_results,
        use_reranking=True,
        use_compression=True,
        use_query_rewrite=True
    )


search_rag_database_tool = tool(
    search_rag_database,
    description="Search RAG database with automatic reranking and compression for better results"
)


# Additional tool for fine-tuned search control
enhanced_search_tool = tool(
    enhanced_search_rag_database,
    description="Advanced search with control over reranking, compression, and query rewriting"
)


def check_rag_for_topic(keywords: str) -> str:
    """Checks if papers on this topic exist in the RAG database."""
    try:
        collection = vectorstore._collection
        count = collection.count()
        
        if count == 0:
            return "Database is empty. No papers stored yet."
        
        results = vectorstore.similarity_search(keywords, k=5)
        
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
    
    Args:
        identifier: Either a PMID (or comma-separated PMIDs) or search keywords
        search_type: "pmid", "query", or "auto" (auto-detect)
    
    Returns:
        Summary of deletion operation
    """
    print(f"Attempting to remove from database: '{identifier}' (type: {search_type})")
    
    # Auto-detect if this is a PMID
    if search_type == "auto":
        clean_id = identifier.replace("PMID:", "").replace("pmid:", "").strip()
        if all(part.strip().isdigit() for part in clean_id.split(',')):
            search_type = "pmid"
        else:
            search_type = "query"
    
    try:
        collection = vectorstore._collection
        
        if search_type == "pmid":
            # Remove by PMID(s)
            pmids = [p.strip() for p in identifier.replace("PMID:", "").replace("pmid:", "").strip().split(',')]
            total_deleted = 0
            
            for pmid in pmids:
                # Get all documents with this PMID
                results = collection.get(
                    where={"pmid": pmid}
                )
                
                if results and results['ids']:
                    # Delete all chunks for this PMID
                    collection.delete(ids=results['ids'])
                    deleted_count = len(results['ids'])
                    total_deleted += deleted_count
                    print(f"   Deleted {deleted_count} chunks for PMID: {pmid}")
                else:
                    print(f"   No documents found for PMID: {pmid}")
            
            if total_deleted > 0:
                return f"Successfully deleted {total_deleted} chunks from {len(pmids)} PMID(s)"
            else:
                return f"No documents found for PMIDs: {', '.join(pmids)}"
                
        elif search_type == "query":
            # Remove by similarity search
            # First, find relevant documents
            results = vectorstore.similarity_search(identifier, k=20)
            
            if not results:
                return f"No documents found matching query: '{identifier}'"
            
            # Group by PMID to show what will be deleted
            pmids_to_delete = {}
            for doc in results:
                pmid = doc.metadata.get("pmid", "unknown")
                title = doc.metadata.get("title", "Unknown")
                if pmid not in pmids_to_delete:
                    pmids_to_delete[pmid] = title
            
            # Confirm deletion
            output = f"Found {len(pmids_to_delete)} paper(s) matching '{identifier}':\n"
            for pmid, title in pmids_to_delete.items():
                output += f"  - PMID: {pmid} | {title[:60]}...\n"
            
            # Delete all chunks for these PMIDs
            total_deleted = 0
            for pmid in pmids_to_delete.keys():
                results = collection.get(
                    where={"pmid": pmid}
                )
                if results and results['ids']:
                    collection.delete(ids=results['ids'])
                    total_deleted += len(results['ids'])
            
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
        
        # Sample to get statistics
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
        
        output += f"└─ Enhanced features: ✓ Reranking ✓ Compression ✓ Query Rewriting\n"
        
        if unique_titles:
            output += f"\nSample papers:\n"
            for title in list(unique_titles)[:3]:
                output += f"  • {title}...\n"
        
        return output
    except Exception as e:
        return f"Error getting stats: {e}"


get_database_stats_tool = tool(get_database_stats)

# ============================================
# ENHANCED AGENT SETUP
# ============================================

memory = MemorySaver()

# Enhanced system prompt
system_prompt = '''You are an advanced PubMed research assistant with enhanced RAG capabilities including:
- Query rewriting for better retrieval
- Cross-encoder reranking for relevance
- Semantic compression for focused content

ENHANCED FEATURES:
1. **Query Rewriting**: Automatically generates multiple query variations to improve retrieval
2. **Reranking**: Uses cross-encoder models to reorder results by relevance
3. **Compression**: Extracts only the most relevant portions of documents

IMPORTANT PAPER RETRIEVAL RULES:

1. SPECIFIC PAPER REQUESTS:
   - When user asks about a SPECIFIC paper (by PMID or title), retrieve ONLY that paper
   - Use the exact PMID if provided (e.g., "tell me about PMID 12345" → search "12345")
   - Use the paper title if that's what they reference
   - The search will return COMPLETE content for single papers

2. MULTIPLE PAPERS:
   - When user asks about a topic, retrieve multiple relevant papers
   - Enhanced search will automatically rerank and compress for better results
   - Provide summaries and comparisons across papers

3. SEARCH STRATEGIES:
   - For "tell me more about [specific paper]": Use PMID or title for targeted retrieval
   - For "what does [paper] say about X": First retrieve the specific paper by PMID
   - For general topics: Use keyword search with automatic enhancement

ENHANCED SEARCH CAPABILITIES:
- The search tool now automatically applies:
  * Query rewriting to find more relevant papers
  * Reranking to prioritize most relevant results
  * Compression to focus on query-relevant content
- You can use 'enhanced_search_tool' for fine control over these features

DELETION RULES:
- When asked to remove/delete papers, use 'remove_from_rag_tool'
- Can delete by PMID (preferred) or by search query
- Always confirm what will be deleted before proceeding
- Inform user of deletion results

WORKFLOW:
1. Check local database with 'check_rag_for_topic_tool'
2. For existing papers:
   - Use 'search_rag_database_tool' (automatically enhanced)
   - Or use 'enhanced_search_tool' for manual control
3. For new searches:
   - Use 'pubmed_search_and_store_tool' first
   - Then retrieve with enhanced search
   - Ask the user for next steps

CRITICAL RULES:
- When discussing a specific paper, cite ONLY from that paper's content
- Never mix information from different papers unless explicitly comparing
- Always use PMIDs when referring to specific papers
- When a user asks for "more details" about a paper, retrieve it by PMID
- Note when results have been reranked or compressed for transparency

RESPONSE FORMAT:
- For single paper: Provide comprehensive analysis using all available content
- For multiple papers: Provide structured comparisons with relevance scores
- Always cite with format: "According to [Title] (PMID: [number])..."
- Mention if results were enhanced (reranked/compressed) for transparency
- Never make up information. Include only information retrieved from papers in answers'''

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
    print("PubMed Research Assistant")
    print("Commands: 'exit' to quit, 'stats' for database info, 'clear' to clear screen\n")
    
    stats_result = get_database_stats()
    print(f"{stats_result}\n")
    
    config = {"configurable": {"thread_id": "cli-thread"}, "recursion_limit":250}
    
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
            
            # Test command for direct multi-year search
            if user_input.lower().startswith("test multi"):
                parts = user_input[10:].strip().split()
                keywords = parts[0] if parts else "CRISPR"
                years = parts[1] if len(parts) > 1 else "2022-2024"
                print(f"\nTesting multi-year search: '{keywords}' for years {years}...")
                result = pubmed_search_and_store_multi_year(keywords, years, pnum=3)
                print(result)
                print(f"\n{get_database_stats()}\n")
                continue
            
            # Regular agent interaction
            print("\nProcessing...")
            
            # Invoke agent with increased recursion limit for complex multi-year searches
            result = pubmed_agent.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config={"configurable": {"thread_id": "cli-thread"}, "recursion_limit": 250}
            )
            
            # Print response
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
