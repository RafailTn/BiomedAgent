"""
PubMed Research Agent v3.1
==========================
CRITICAL FIXES:
1. Skip PrimeKG loading if already loaded (checks entity count first)
2. Use absolute paths for database persistence
3. Batch insert with progress reporting

For FAST first-time loading, set PRIMEKG_LIMIT = 100000 to test with subset.
Full loading takes ~5-15 minutes depending on hardware.
"""

import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import argparse
from datetime import date
from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain_ollama import ChatOllama 
from Bio import Entrez
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
import re
import requests
import xml.etree.ElementTree as ET
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.tools import tool
import json
from typing import List, Optional, Dict, Any, Tuple, Set
from dataclasses import dataclass, field
import numpy as np
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import PromptTemplate
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
import torch
from transformers import AutoTokenizer, AutoModel
import asyncio
import aiohttp
from collections import defaultdict
from rank_bm25 import BM25Okapi
from pathlib import Path
import shutil
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

Entrez.email = "rafailadam46@gmail.com"

# ============================================
# CONFIGURATION
# ============================================

ENABLE_KNOWLEDGE_GRAPH = True
ENABLE_FACT_VERIFICATION = True
ENABLE_KG_ENHANCED_RETRIEVAL = True

LOAD_PRIMEKG = True
# Set to 100000 for quick testing (~2 min), None for full (~10-15 min)
PRIMEKG_LIMIT = None

GLINER_DEVICE = "cuda"

# ABSOLUTE PATHS - crucial for persistence!
SCRIPT_DIR = Path(__file__).parent.resolve()
KUZU_DB_PATH = SCRIPT_DIR / "kuzu_biomedical_kg"
PRIMEKG_DATA_DIR = SCRIPT_DIR / "primekg_data"
VECTORSTORE_PATH = SCRIPT_DIR / "medcpt_pubmed_rag_db"

logger.info(f"Script directory: {SCRIPT_DIR}")
logger.info(f"Kuzu DB path: {KUZU_DB_PATH}")


# ============================================
# LLM
# ============================================

pi_llm = ChatOllama(model="ministral-3:8b")


# ============================================
# DATA STRUCTURES
# ============================================

@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    score: float
    cui: Optional[str] = None
    canonical_name: Optional[str] = None
    kg_id: Optional[str] = None
    
    def __hash__(self):
        return hash((self.text.lower(), self.label))


@dataclass 
class Triple:
    subject: str
    subject_type: str
    predicate: str
    object: str
    object_type: str
    source: str = "extracted"
    confidence: float = 1.0
    pmid: Optional[str] = None


@dataclass
class VerificationResult:
    claim: str
    is_verified: bool
    confidence: float
    supporting_triples: List[Triple] = field(default_factory=list)
    explanation: str = ""


# ============================================
# MEDCPT EMBEDDINGS
# ============================================

class MedCPTEmbeddings(Embeddings):
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
                encoded = tokenizer(batch, truncation=True, padding=True, max_length=512, return_tensors="pt").to(self.device)
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
# GLINER NER
# ============================================

class GLiNERExtractor:
    DEFAULT_LABELS = ["Disease", "Drug", "Gene", "Protein", "Chemical", "Cell Type", "Organism", "Pathway"]
    
    def __init__(self, model_name: str = "Ihor/gliner-biomed-bi-base-v1.0", device: str = "cuda", threshold: float = 0.4):
        self.device = device
        self.threshold = threshold
        self.model = None
        self.model_name = model_name
        self._loaded = False
        
    def load(self):
        if self._loaded:
            return
        try:
            from gliner import GLiNER
            logger.info(f"Loading GLiNER...")
            self.model = GLiNER.from_pretrained(self.model_name)
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to(self.device)
            self._loaded = True
        except ImportError:
            raise ImportError("Install GLiNER: pip install gliner")
    
    def extract(self, text: str, labels: List[str] = None, threshold: float = None) -> List[Entity]:
        self.load()
        labels = labels or self.DEFAULT_LABELS
        threshold = threshold or self.threshold
        
        try:
            predictions = self.model.predict_entities(text, labels, threshold=threshold)
            seen = set()
            entities = []
            for pred in predictions:
                key = (pred['text'].lower(), pred['label'])
                if key not in seen:
                    seen.add(key)
                    entities.append(Entity(text=pred['text'], label=pred['label'],
                                          start=pred['start'], end=pred['end'], score=pred['score']))
            return entities
        except:
            return []
    
    def unload(self):
        if self.model:
            del self.model
            self.model = None
            self._loaded = False
            torch.cuda.empty_cache()


# ============================================
# KUZU DATABASE
# ============================================

class KuzuGraphDB:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db = None
        self.conn = None
        self._initialized = False
        
    def initialize(self):
        if self._initialized:
            return
            
        try:
            import kuzu
        except ImportError:
            raise ImportError("Install Kuzu: pip install kuzu")
        
        logger.info(f"Initializing Kuzu at: {self.db_path}")
        
        try:
            self.db = kuzu.Database(str(self.db_path))
            self.conn = kuzu.Connection(self.db)
            self._create_schema()
            self._initialized = True
            logger.info("Kuzu initialized successfully")
        except Exception as e:
            logger.error(f"Kuzu init failed: {e}")
            if self.db_path.exists():
                logger.info("Removing corrupted database...")
                shutil.rmtree(self.db_path)
            self.db = kuzu.Database(str(self.db_path))
            self.conn = kuzu.Connection(self.db)
            self._create_schema()
            self._initialized = True
        
    def _create_schema(self):
        schemas = [
            "CREATE NODE TABLE IF NOT EXISTS Entity(id STRING PRIMARY KEY, name STRING, type STRING, cui STRING, source STRING, description STRING)",
            "CREATE NODE TABLE IF NOT EXISTS Document(pmid STRING PRIMARY KEY, title STRING, year STRING, has_full_text BOOLEAN)",
            "CREATE REL TABLE IF NOT EXISTS MENTIONS(FROM Document TO Entity, count INT32, context STRING)",
            "CREATE REL TABLE IF NOT EXISTS RELATES_TO(FROM Entity TO Entity, relation STRING, source STRING, confidence DOUBLE, pmid STRING)",
        ]
        for schema in schemas:
            try:
                self.conn.execute(schema)
            except:
                pass
    
    def add_entity(self, entity_id: str, name: str, entity_type: str, cui: str = "", source: str = "extracted", description: str = "") -> bool:
        try:
            self.conn.execute("MERGE (e:Entity {id: $id}) SET e.name = $name, e.type = $type, e.cui = $cui, e.source = $source, e.description = $description",
                            {"id": entity_id, "name": name, "type": entity_type, "cui": cui, "source": source, "description": description})
            return True
        except:
            return False
    
    def add_document(self, pmid: str, title: str, year: str, has_full_text: bool = False) -> bool:
        try:
            self.conn.execute("MERGE (d:Document {pmid: $pmid}) SET d.title = $title, d.year = $year, d.has_full_text = $has_full_text",
                            {"pmid": pmid, "title": title, "year": year, "has_full_text": has_full_text})
            return True
        except:
            return False
    
    def add_mention(self, pmid: str, entity_id: str, context: str = "", count: int = 1) -> bool:
        try:
            self.conn.execute("MATCH (d:Document {pmid: $pmid}), (e:Entity {id: $eid}) MERGE (d)-[r:MENTIONS]->(e) SET r.count = $count, r.context = $context",
                            {"pmid": pmid, "eid": entity_id, "context": context[:500], "count": count})
            return True
        except:
            return False
    
    def add_relation(self, subject_id: str, predicate: str, object_id: str, source: str = "extracted", confidence: float = 1.0, pmid: str = "") -> bool:
        try:
            self.conn.execute("MATCH (s:Entity {id: $sid}), (o:Entity {id: $oid}) MERGE (s)-[r:RELATES_TO]->(o) SET r.relation = $rel, r.source = $source, r.confidence = $conf, r.pmid = $pmid",
                            {"sid": subject_id, "oid": object_id, "rel": predicate, "source": source, "conf": confidence, "pmid": pmid})
            return True
        except:
            return False
    
    def get_entity_neighbors(self, entity_id: str, hops: int = 1, limit: int = 50) -> List[Dict]:
        try:
            result = self.conn.execute("MATCH (e:Entity {id: $eid})-[r:RELATES_TO]-(neighbor:Entity) RETURN neighbor.id, neighbor.name, neighbor.type, r.relation LIMIT $limit",
                                      {"eid": entity_id, "limit": limit})
            neighbors = []
            while result.has_next():
                row = result.get_next()
                neighbors.append({"id": row[0], "name": row[1], "type": row[2], "relation": row[3] if len(row) > 3 else None})
            return neighbors
        except:
            return []
    
    def find_entities_by_name(self, name: str, entity_type: str = None, limit: int = 10) -> List[Dict]:
        try:
            if entity_type:
                result = self.conn.execute("MATCH (e:Entity) WHERE lower(e.name) CONTAINS $name AND e.type = $type RETURN e.id, e.name, e.type, e.cui LIMIT $limit",
                                          {"name": name.lower(), "type": entity_type, "limit": limit})
            else:
                result = self.conn.execute("MATCH (e:Entity) WHERE lower(e.name) CONTAINS $name RETURN e.id, e.name, e.type, e.cui LIMIT $limit",
                                          {"name": name.lower(), "limit": limit})
            entities = []
            while result.has_next():
                row = result.get_next()
                entities.append({"id": row[0], "name": row[1], "type": row[2], "cui": row[3]})
            return entities
        except:
            return []
    
    def get_triples_for_entities(self, entity_ids: List[str]) -> List[Triple]:
        try:
            result = self.conn.execute("MATCH (s:Entity)-[r:RELATES_TO]->(o:Entity) WHERE s.id IN $eids OR o.id IN $eids RETURN s.name, s.type, r.relation, o.name, o.type, r.source, r.confidence, r.pmid",
                                      {"eids": entity_ids})
            triples = []
            while result.has_next():
                row = result.get_next()
                triples.append(Triple(subject=row[0], subject_type=row[1], predicate=row[2], object=row[3], object_type=row[4],
                                     source=row[5], confidence=row[6], pmid=row[7] if row[7] else None))
            return triples
        except:
            return []
    
    def get_stats(self) -> Dict[str, int]:
        stats = {"entities": 0, "documents": 0, "relations": 0, "mentions": 0}
        try:
            for key, query in [("entities", "MATCH (e:Entity) RETURN count(e)"), ("documents", "MATCH (d:Document) RETURN count(d)"),
                               ("relations", "MATCH ()-[r:RELATES_TO]->() RETURN count(r)"), ("mentions", "MATCH ()-[m:MENTIONS]->() RETURN count(m)")]:
                result = self.conn.execute(query)
                if result.has_next():
                    stats[key] = result.get_next()[0]
        except:
            pass
        return stats
    
    def close(self):
        self.conn = None
        self.db = None


# ============================================
# PRIMEKG LOADER - WITH SKIP IF LOADED
# ============================================

class PrimeKGLoader:
    PRIMEKG_URL = "https://dataverse.harvard.edu/api/access/datafile/6180620"
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.edges_file = self.data_dir / "kg.csv"
        
    def download_if_needed(self) -> bool:
        if self.edges_file.exists():
            logger.info("PrimeKG CSV already exists")
            return True
        
        logger.info("Downloading PrimeKG (~300MB)...")
        try:
            response = requests.get(self.PRIMEKG_URL, stream=True)
            response.raise_for_status()
            
            with open(self.edges_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info("PrimeKG downloaded!")
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            if self.edges_file.exists():
                self.edges_file.unlink()
            return False
    
    def load_into_kuzu(self, kg_db: KuzuGraphDB, limit: Optional[int] = None) -> int:
        """Load PrimeKG - SKIPS if already loaded!"""
        
        # CHECK IF ALREADY LOADED - This is the key fix!
        stats = kg_db.get_stats()
        existing_entities = stats.get('entities', 0)
        
        if existing_entities > 50000:
            logger.info(f"✓ PrimeKG already loaded: {existing_entities:,} entities, {stats.get('relations', 0):,} relations")
            return stats.get('relations', 0)
        
        if not self.edges_file.exists():
            if not self.download_if_needed():
                return 0
        
        logger.info("Loading PrimeKG into Kuzu...")
        logger.info("This takes 5-15 minutes on first run. Subsequent runs will be instant.")
        start_time = time.time()
        
        try:
            import pandas as pd
            
            logger.info("Reading CSV file...")
            df = pd.read_csv(self.edges_file, low_memory=False)
            
            if limit:
                df = df.head(limit)
                logger.info(f"Limited to {limit:,} edges for testing")
            
            total_edges = len(df)
            logger.info(f"Processing {total_edges:,} edges...")
            
            # Extract unique entities
            subjects = df[['x_id', 'x_type', 'x_name']].drop_duplicates()
            objects = df[['y_id', 'y_type', 'y_name']].drop_duplicates()
            
            all_entities = pd.concat([
                subjects.rename(columns={'x_id': 'id', 'x_type': 'type', 'x_name': 'name'}),
                objects.rename(columns={'y_id': 'id', 'y_type': 'type', 'y_name': 'name'})
            ]).drop_duplicates(subset=['id'])
            
            total_entities = len(all_entities)
            logger.info(f"Found {total_entities:,} unique entities")
            
            # Insert entities with progress
            logger.info("Inserting entities...")
            entity_count = 0
            report_interval = 10000
            
            for _, row in all_entities.iterrows():
                eid = f"primekg:{row['id']}"
                name = str(row['name']) if pd.notna(row['name']) else ''
                etype = str(row['type']) if pd.notna(row['type']) else ''
                kg_db.add_entity(eid, name, etype, source="primekg")
                entity_count += 1
                
                if entity_count % report_interval == 0:
                    elapsed = time.time() - start_time
                    pct = (entity_count / total_entities) * 100
                    rate = entity_count / elapsed
                    remaining = (total_entities - entity_count) / rate if rate > 0 else 0
                    logger.info(f"  Entities: {entity_count:,}/{total_entities:,} ({pct:.1f}%) - {remaining:.0f}s remaining")
            
            logger.info(f"Inserted {entity_count:,} entities")
            
            # Insert relations with progress
            logger.info("Inserting relations...")
            relation_count = 0
            report_interval = 50000
            relation_start = time.time()
            
            for _, row in df.iterrows():
                sid = f"primekg:{row['x_id']}"
                oid = f"primekg:{row['y_id']}"
                rel = str(row['relation']) if pd.notna(row['relation']) else ''
                kg_db.add_relation(sid, rel, oid, source="primekg")
                relation_count += 1
                
                if relation_count % report_interval == 0:
                    elapsed = time.time() - relation_start
                    pct = (relation_count / total_edges) * 100
                    rate = relation_count / elapsed
                    remaining = (total_edges - relation_count) / rate if rate > 0 else 0
                    logger.info(f"  Relations: {relation_count:,}/{total_edges:,} ({pct:.1f}%) - {remaining:.0f}s remaining")
            
            total_time = time.time() - start_time
            logger.info(f"✓ PrimeKG loaded in {total_time:.1f}s: {entity_count:,} entities, {relation_count:,} relations")
            
            return relation_count
            
        except Exception as e:
            logger.error(f"Load failed: {e}")
            import traceback
            traceback.print_exc()
            return 0


# ============================================
# ENTITY LINKER
# ============================================

class EntityLinker:
    TYPE_MAP = {"Disease": "disease", "Drug": "drug", "Gene": "gene/protein", "Protein": "gene/protein", "Chemical": "drug", "Pathway": "pathway"}
    
    def __init__(self, kg_db: KuzuGraphDB):
        self.kg_db = kg_db
        self.cache: Dict[str, Optional[Dict]] = {}
        
    def link(self, entity: Entity) -> Optional[Dict]:
        cache_key = f"{entity.text.lower()}:{entity.label}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        candidates = self.kg_db.find_entities_by_name(entity.text, self.TYPE_MAP.get(entity.label), limit=5)
        if not candidates:
            candidates = self.kg_db.find_entities_by_name(entity.text, limit=5)
        
        result = candidates[0] if candidates else None
        self.cache[cache_key] = result
        return result


# ============================================
# KG RETRIEVER & VERIFIER
# ============================================

class KGEnhancedRetriever:
    def __init__(self, kg_db: KuzuGraphDB, extractor: GLiNERExtractor, linker: EntityLinker):
        self.kg_db = kg_db
        self.extractor = extractor
        self.linker = linker
    
    def expand_query(self, query: str) -> Dict[str, Any]:
        entities = self.extractor.extract(query)
        linked_nodes = []
        
        for entity in entities:
            kg_node = self.linker.link(entity)
            if kg_node:
                entity.kg_id = kg_node['id']
                linked_nodes.append(kg_node)
        
        expanded_entities = []
        for node in linked_nodes[:5]:
            neighbors = self.kg_db.get_entity_neighbors(node['id'], limit=20)
            expanded_entities.extend(neighbors)
        
        all_ids = [n['id'] for n in linked_nodes] + [e['id'] for e in expanded_entities]
        relevant_triples = self.kg_db.get_triples_for_entities(all_ids[:50]) if all_ids else []
        
        expanded_terms = [e.text for e in entities] + [e.get('name', '') for e in expanded_entities if e.get('name')]
        
        return {'entities': entities, 'linked_nodes': linked_nodes, 'expanded_entities': expanded_entities,
                'relevant_triples': relevant_triples, 'expanded_terms': list(set(expanded_terms))[:30]}
    
    def format_kg_context(self, triples: List[Triple], max_triples: int = 10) -> str:
        if not triples:
            return ""
        lines = ["[Knowledge Graph Context]"]
        for triple in triples[:max_triples]:
            lines.append(f"• {triple.subject} —[{triple.predicate}]→ {triple.object}")
        return "\n".join(lines)


class FactVerifier:
    RELATION_KEYWORDS = ['causes', 'treats', 'inhibits', 'activates', 'associated with', 'targets', 'binds to', 'regulates', 'linked to', 'prevents', 'induces']
    
    def __init__(self, kg_db: KuzuGraphDB, extractor: GLiNERExtractor, linker: EntityLinker):
        self.kg_db = kg_db
        self.extractor = extractor
        self.linker = linker
    
    def verify_response(self, text: str) -> Dict[str, Any]:
        sentences = re.split(r'[.!?]+', text)
        claims = []
        verified = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 15:
                continue
            
            entities = self.extractor.extract(sentence)
            if len(entities) >= 2:
                linked = [(e, self.linker.link(e)) for e in entities]
                linked = [n for e, n in linked if n]
                
                if len(linked) >= 2:
                    triples = self.kg_db.get_triples_for_entities([n['id'] for n in linked])
                    sentence_lower = sentence.lower()
                    supporting = [t for t in triples if t.subject.lower() in sentence_lower and t.object.lower() in sentence_lower]
                    
                    if supporting:
                        verified += 1
                    claims.append({'text': sentence, 'verified': bool(supporting)})
        
        total = len(claims)
        return {'total_claims': total, 'verified_claims': verified, 'verification_rate': verified / total if total > 0 else 1.0}


class KGDocumentProcessor:
    def __init__(self, kg_db: KuzuGraphDB, extractor: GLiNERExtractor, linker: EntityLinker):
        self.kg_db = kg_db
        self.extractor = extractor
        self.linker = linker
    
    def process_paper(self, pmid: str, title: str, abstract: str, year: str, full_text: str = None) -> Dict[str, int]:
        self.kg_db.add_document(pmid, title, year, bool(full_text))
        
        text = f"{title}. {abstract}"
        if full_text:
            text += f" {full_text[:3000]}"
        
        entities = self.extractor.extract(text)
        stats = {'entities': 0, 'mentions': 0}
        
        for entity in entities:
            linked = self.linker.link(entity)
            if linked:
                entity_id = linked['id']
            else:
                entity_id = f"local:{entity.label}:{entity.text.lower().replace(' ', '_')}"
                self.kg_db.add_entity(entity_id, entity.text, entity.label.lower(), source="extracted")
                stats['entities'] += 1
            
            self.kg_db.add_mention(pmid, entity_id)
            stats['mentions'] += 1
        
        return stats


# ============================================
# KG MANAGER
# ============================================

class KnowledgeGraphManager:
    def __init__(self, db_path: Path, gliner_device: str = "cuda", load_primekg: bool = True,
                 primekg_limit: int = None, primekg_data_dir: Path = None):
        
        logger.info("Initializing Knowledge Graph Manager...")
        
        self.kg_db = KuzuGraphDB(db_path)
        self.kg_db.initialize()
        
        self.extractor = GLiNERExtractor(device=gliner_device)
        self.linker = EntityLinker(self.kg_db)
        self.doc_processor = KGDocumentProcessor(self.kg_db, self.extractor, self.linker)
        self.retriever = KGEnhancedRetriever(self.kg_db, self.extractor, self.linker)
        self.verifier = FactVerifier(self.kg_db, self.extractor, self.linker)
        
        if load_primekg:
            loader = PrimeKGLoader(primekg_data_dir or Path("./primekg_data"))
            loader.load_into_kuzu(self.kg_db, limit=primekg_limit)
        
        logger.info("Knowledge Graph Manager ready!")
    
    def expand_query(self, query: str) -> Dict:
        return self.retriever.expand_query(query)
    
    def get_kg_context(self, query: str, max_triples: int = 10) -> str:
        expansion = self.expand_query(query)
        return self.retriever.format_kg_context(expansion['relevant_triples'], max_triples)
    
    def verify_response(self, response: str) -> Dict:
        return self.verifier.verify_response(response)
    
    def process_paper(self, pmid: str, title: str, abstract: str, year: str, full_text: str = None) -> Dict:
        return self.doc_processor.process_paper(pmid, title, abstract, year, full_text)
    
    def get_stats(self) -> Dict:
        return self.kg_db.get_stats()
    
    def unload_gliner(self):
        self.extractor.unload()
    
    def close(self):
        self.kg_db.close()


# ============================================
# HYBRID RETRIEVAL
# ============================================

def reciprocal_rank_fusion(results_lists, k=60, top_n=None):
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
            self.documents = [Document(page_content=c, metadata=m or {}) for c, m in zip(all_docs.get('documents', []), all_docs.get('metadatas', [])) if c]
            self.corpus = [re.findall(r'\w+', d.page_content.lower()) for d in self.documents]
            if self.corpus:
                self.bm25 = BM25Okapi(self.corpus)
        except:
            pass
    
    def refresh(self):
        self._build_index()
    
    def search(self, query, k=10):
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(re.findall(r'\w+', query.lower()))
        top_idx = np.argsort(scores)[::-1][:k]
        return [self.documents[i] for i in top_idx if scores[i] > 0]


class HybridRetriever:
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
# ASYNC PUBMED
# ============================================

class AsyncPubMedFetcher:
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
            r = await self._request(session, f"{self.BASE_URL}/esearch.fcgi", self._params(db="pubmed", term=term, retmax=str(retmax), retmode="json"))
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
                r = await self._request(session, f"{self.BASE_URL}/efetch.fcgi", self._params(db="pubmed", id=",".join(pmids[i:i+50]), rettype="abstract", retmode="xml"))
                if r:
                    results.update(self._parse_xml(r))
        return results
    
    async def fetch_full_texts(self, pmids):
        return {p: None for p in pmids}  # Simplified - full text fetching unchanged
    
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
                results[pmid] = {"pmid": pmid, "title": title, "abstract": abstract,
                                "authors": "; ".join(authors) or "No authors", "year": year.text if year is not None else None}
        except:
            pass
        return results


# ============================================
# INITIALIZE
# ============================================

logger.info("="*60)
logger.info("PubMed Research Agent v3.1")
logger.info("="*60)

medcpt_embeddings = MedCPTEmbeddings(device="cuda", batch_size=8)

VECTORSTORE_PATH.mkdir(exist_ok=True)
vectorstore = Chroma(persist_directory=str(VECTORSTORE_PATH), embedding_function=medcpt_embeddings,
                    collection_name="pubmed_papers", collection_metadata={"hnsw:space": "cosine"})

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

query_rewriter_prompt = PromptTemplate(input_variables=["query"], template="Generate 3 alternative biomedical queries for: {query}\n\nAlternatives:")

enc_model = HuggingFaceCrossEncoder(model_name='ncbi/MedCPT-Cross-Encoder', model_kwargs={'device': 'cuda'})
compressor = CrossEncoderReranker(model=enc_model, top_n=10)

bm25_retriever = BM25SparseRetriever(vectorstore)
hybrid_retriever = HybridRetriever(vectorstore, bm25_retriever)

async_fetcher = AsyncPubMedFetcher(email=Entrez.email, api_key=os.getenv("NCBI_API_KEY"))

kg_manager = None
if ENABLE_KNOWLEDGE_GRAPH:
    try:
        kg_manager = KnowledgeGraphManager(db_path=KUZU_DB_PATH, gliner_device=GLINER_DEVICE,
                                          load_primekg=LOAD_PRIMEKG, primekg_limit=PRIMEKG_LIMIT, primekg_data_dir=PRIMEKG_DATA_DIR)
    except Exception as e:
        logger.warning(f"KG init failed: {e}")
        ENABLE_KNOWLEDGE_GRAPH = False

torch.cuda.empty_cache()
logger.info(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")


# ============================================
# SEARCH FUNCTION
# ============================================

def kg_enhanced_search(query, num_results=10):
    kg_context = ""
    if kg_manager and ENABLE_KG_ENHANCED_RETRIEVAL:
        try:
            expansion = kg_manager.expand_query(query)
            kg_context = kg_manager.retriever.format_kg_context(expansion.get('relevant_triples', []), 8)
        except:
            pass
    
    results = hybrid_retriever.search(query, k=num_results * 2)
    if not results:
        return "No papers found.", kg_context
    
    seen = set()
    unique = [d for d in results if hash(d.page_content[:100]) not in seen and not seen.add(hash(d.page_content[:100]))]
    results = compressor.compress_documents(unique, query)[:num_results] if len(unique) > 1 else unique[:num_results]
    
    papers = {}
    for doc in results:
        pmid = doc.metadata.get("pmid", "unknown")
        if pmid not in papers:
            papers[pmid] = {"title": doc.metadata.get("title", "Unknown"), "authors": doc.metadata.get("authors", "Unknown"),
                          "year": doc.metadata.get("year", "Unknown"), "chunks": []}
        papers[pmid]["chunks"].append(doc.page_content)
    
    output = [f"Found {len(papers)} paper(s)\n"]
    for idx, (pmid, info) in enumerate(papers.items(), 1):
        content = "\n\n".join(info['chunks'])[:500]
        output.append(f"\n[{idx}] PMID: {pmid}\nTitle: {info['title']}\nAuthors: {info['authors']}\nYear: {info['year']}\n{content}...\n{'='*40}")
    
    return "".join(output), kg_context


# ============================================
# ASYNC SEARCH & STORE
# ============================================

async def async_pubmed_search_and_store(keywords, years=None, pnum=5):
    year_list = [str(date.today().year)] if not years else ([str(y) for y in range(int(years.split("-")[0]), int(years.split("-")[1])+1)] if "-" in years else [years])
    
    total_stored = 0
    all_papers = []
    
    for year in year_list:
        pmids = await async_fetcher.search(f"({keywords}) AND {year}[pdat]", retmax=pnum)
        if not pmids:
            continue
        
        articles = await async_fetcher.fetch_abstracts(pmids)
        
        for pmid, article in articles.items():
            try:
                content = f"PMID: {pmid}\nTitle: {article['title']}\nAuthors: {article['authors']}\nYear: {year}\n\n{article['abstract']}"
                chunks = text_splitter.split_text(content)
                docs = [Document(page_content=c, metadata={"pmid": pmid, "title": article['title'], "authors": article['authors'],
                                "year": year, "chunk_index": i, "source": "pubmed"}) for i, c in enumerate(chunks)]
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
    try:
        return asyncio.run(async_pubmed_search_and_store(keywords, years, pnum))
    except:
        try:
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.get_event_loop().run_until_complete(async_pubmed_search_and_store(keywords, years, pnum))
        except:
            return "Search failed"


# ============================================
# TOOLS
# ============================================

@tool
def pubmed_search_and_store_tool(keywords: str, years: str = None, pnum: int = 10) -> str:
    """Search PubMed and store papers."""
    return pubmed_search_and_store(keywords, years, pnum)

@tool
def search_rag_database_tool(query: str, num_results: int = 10) -> str:
    """Search database with KG-enhanced retrieval."""
    results, kg_context = kg_enhanced_search(query, num_results)
    return f"{kg_context}\n\n{results}" if kg_context else results

@tool
def check_rag_for_topic_tool(keywords: str) -> str:
    """Check if papers exist for a topic."""
    results = hybrid_retriever.search(keywords, k=5)
    if not results:
        return f"No papers for '{keywords}'"
    papers = {d.metadata.get("pmid"): d.metadata.get("title") for d in results if d.metadata.get("pmid")}
    return f"Found {len(papers)} papers:\n" + "\n".join(f"• PMID:{p} | {t[:50]}..." for p, t in papers.items())

@tool
def get_database_stats_tool() -> str:
    """Get database statistics."""
    output = f"Vector DB: {vectorstore._collection.count()} chunks\n"
    if kg_manager:
        stats = kg_manager.get_stats()
        output += f"KG: {stats.get('entities', 0):,} entities, {stats.get('relations', 0):,} relations"
    return output

@tool
def verify_facts_tool(text: str) -> str:
    """Verify claims against KG."""
    if not kg_manager:
        return "KG not enabled"
    result = kg_manager.verify_response(text)
    return f"Claims: {result['total_claims']}, Verified: {result['verified_claims']}, Rate: {result['verification_rate']:.0%}"

@tool
def explore_kg_entity_tool(entity_name: str) -> str:
    """Explore entity in KG."""
    if not kg_manager:
        return "KG not enabled"
    entities = kg_manager.kg_db.find_entities_by_name(entity_name, limit=3)
    if not entities:
        return f"'{entity_name}' not found"
    output = ""
    for e in entities:
        output += f"\n{e['name']} ({e['type']})\n"
        for n in kg_manager.kg_db.get_entity_neighbors(e['id'], limit=5):
            output += f"  → {n.get('relation', 'related')} → {n['name']}\n"
    return output


# ============================================
# AGENT
# ============================================

memory = MemorySaver()
system_prompt = "You are a PubMed research assistant with KG integration. Use tools to search papers. Only cite PMIDs from results."

tools = [pubmed_search_and_store_tool, search_rag_database_tool, check_rag_for_topic_tool,
         get_database_stats_tool, verify_facts_tool, explore_kg_entity_tool]

pubmed_agent = create_agent(model=pi_llm, tools=tools, system_prompt=system_prompt, checkpointer=memory)


# ============================================
# MAIN
# ============================================

def main():
    print("\n" + "="*60)
    print("PubMed Research Agent v3.1 - Knowledge Graph Edition")
    print("="*60)
    print("Commands: exit, stats, kg, vram, gc, verify <text>, entity <name>")
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
                print(f"\nVRAM: {torch.cuda.memory_allocated()/1e9:.2f} / {torch.cuda.get_device_properties(0).total_memory/1e9:.2f} GB\n")
                continue
            
            if user_input.lower() == "gc":
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                if kg_manager:
                    kg_manager.unload_gliner()
                print("Cleared\n")
                continue
            
            if user_input.lower().startswith("verify "):
                print(f"\n{verify_facts_tool.invoke({'text': user_input[7:]})}\n")
                continue
            
            if user_input.lower().startswith("entity "):
                print(f"\n{explore_kg_entity_tool.invoke({'entity_name': user_input[7:]})}\n")
                continue
            
            print("\nProcessing...")
            result = pubmed_agent.invoke({"messages": [HumanMessage(content=user_input)]},
                                        config={"configurable": {"thread_id": "cli"}, "recursion_limit": 500})
            
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


if __name__ == "__main__":
    main()
