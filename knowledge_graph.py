"""
Knowledge Graph Module for PubMed Research Agent
=================================================
Contains all knowledge graph related classes.

Usage:
    from knowledge_graph import KnowledgeGraphManager
    
    kg_manager = KnowledgeGraphManager(
        db_path="./kuzu_biomedical_kg",
        gliner_device="cpu",
        load_primekg=True
    )
"""

import os
import re
import time
import logging
import shutil
import requests
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================
# DATA STRUCTURES
# ============================================

@dataclass
class Entity:
    """Represents an extracted biomedical entity."""
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
    
    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.text.lower() == other.text.lower() and self.label == other.label


@dataclass 
class Triple:
    """Represents a knowledge graph triple."""
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
    """Result of fact verification."""
    claim: str
    is_verified: bool
    confidence: float
    supporting_triples: List[Triple] = field(default_factory=list)
    explanation: str = ""


# ============================================
# GLINER BIOMEDICAL NER
# ============================================

class GLiNERExtractor:
    """Zero-shot biomedical NER using GLiNER."""
    
    DEFAULT_LABELS = ["Disease", "Drug", "Gene", "Protein", "Chemical", "Cell Type", "Organism", "Pathway"]
    
    def __init__(self, model_name: str = "urchade/gliner_medium_bio-v1.0",
                 device: str = "cpu", threshold: float = 0.4):
        self.device = device
        self.threshold = threshold
        self.model = None
        self.model_name = model_name
        self._loaded = False
        
    def load(self):
        """Load GLiNER model (lazy loading)."""
        if self._loaded:
            return
        
        try:
            from gliner import GLiNER
        except ImportError:
            raise ImportError("GLiNER not installed. Run: pip install gliner --break-system-packages")
        
        logger.info(f"Loading GLiNER on {self.device}...")
        self.model = GLiNER.from_pretrained(self.model_name)
        
        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to(self.device)
        else:
            self.model = self.model.to("cpu")
            
        self._loaded = True
        logger.info("GLiNER loaded")
    
    def extract(self, text: str, labels: List[str] = None, threshold: float = None) -> List[Entity]:
        """Extract biomedical entities from text."""
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
                    entities.append(Entity(
                        text=pred['text'], label=pred['label'],
                        start=pred['start'], end=pred['end'], score=pred['score']
                    ))
            return entities
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []
    
    def unload(self):
        """Unload model to free memory."""
        if self.model:
            del self.model
            self.model = None
            self._loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("GLiNER unloaded")
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ============================================
# KUZU GRAPH DATABASE
# ============================================

class KuzuGraphDB:
    """Embedded graph database using Kuzu."""
    
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db = None
        self.conn = None
        self._initialized = False
        
    def initialize(self):
        """Initialize database and create schema."""
        if self._initialized:
            return
            
        try:
            import kuzu
        except ImportError:
            raise ImportError("Kuzu not installed. Run: pip install kuzu --break-system-packages")
        
        logger.info(f"Initializing Kuzu at: {self.db_path}")
        
        try:
            self.db = kuzu.Database(str(self.db_path))
            self.conn = kuzu.Connection(self.db)
            self._create_schema()
            self._initialized = True
            logger.info("Kuzu initialized")
        except Exception as e:
            logger.error(f"Kuzu init failed: {e}")
            if self.db_path.exists():
                shutil.rmtree(self.db_path)
            self.db = kuzu.Database(str(self.db_path))
            self.conn = kuzu.Connection(self.db)
            self._create_schema()
            self._initialized = True
        
    def _create_schema(self):
        """Create graph schema."""
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
    
    def add_entity(self, entity_id: str, name: str, entity_type: str,
                   cui: str = "", source: str = "extracted", description: str = "") -> bool:
        try:
            self.conn.execute(
                "MERGE (e:Entity {id: $id}) SET e.name = $name, e.type = $type, e.cui = $cui, e.source = $source, e.description = $description",
                {"id": entity_id, "name": name, "type": entity_type, "cui": cui, "source": source, "description": description}
            )
            return True
        except:
            return False
    
    def add_document(self, pmid: str, title: str, year: str, has_full_text: bool = False) -> bool:
        try:
            self.conn.execute(
                "MERGE (d:Document {pmid: $pmid}) SET d.title = $title, d.year = $year, d.has_full_text = $has_full_text",
                {"pmid": pmid, "title": title, "year": year, "has_full_text": has_full_text}
            )
            return True
        except:
            return False
    
    def add_mention(self, pmid: str, entity_id: str, context: str = "", count: int = 1) -> bool:
        try:
            self.conn.execute(
                "MATCH (d:Document {pmid: $pmid}), (e:Entity {id: $eid}) MERGE (d)-[r:MENTIONS]->(e) SET r.count = $count, r.context = $context",
                {"pmid": pmid, "eid": entity_id, "context": context[:500], "count": count}
            )
            return True
        except:
            return False
    
    def add_relation(self, subject_id: str, predicate: str, object_id: str,
                     source: str = "extracted", confidence: float = 1.0, pmid: str = "") -> bool:
        try:
            self.conn.execute(
                "MATCH (s:Entity {id: $sid}), (o:Entity {id: $oid}) MERGE (s)-[r:RELATES_TO]->(o) SET r.relation = $rel, r.source = $source, r.confidence = $conf, r.pmid = $pmid",
                {"sid": subject_id, "oid": object_id, "rel": predicate, "source": source, "conf": confidence, "pmid": pmid}
            )
            return True
        except:
            return False
    
    def get_entity_neighbors(self, entity_id: str, hops: int = 1, limit: int = 50) -> List[Dict]:
        try:
            result = self.conn.execute(
                "MATCH (e:Entity {id: $eid})-[r:RELATES_TO]-(neighbor:Entity) RETURN neighbor.id, neighbor.name, neighbor.type, r.relation LIMIT $limit",
                {"eid": entity_id, "limit": limit}
            )
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
                result = self.conn.execute(
                    "MATCH (e:Entity) WHERE lower(e.name) CONTAINS $name AND e.type = $type RETURN e.id, e.name, e.type, e.cui LIMIT $limit",
                    {"name": name.lower(), "type": entity_type, "limit": limit}
                )
            else:
                result = self.conn.execute(
                    "MATCH (e:Entity) WHERE lower(e.name) CONTAINS $name RETURN e.id, e.name, e.type, e.cui LIMIT $limit",
                    {"name": name.lower(), "limit": limit}
                )
            entities = []
            while result.has_next():
                row = result.get_next()
                entities.append({"id": row[0], "name": row[1], "type": row[2], "cui": row[3]})
            return entities
        except:
            return []
    
    def get_triples_for_entities(self, entity_ids: List[str]) -> List[Triple]:
        if not entity_ids:
            return []
        try:
            result = self.conn.execute(
                "MATCH (s:Entity)-[r:RELATES_TO]->(o:Entity) WHERE s.id IN $eids OR o.id IN $eids RETURN s.name, s.type, r.relation, o.name, o.type, r.source, r.confidence, r.pmid",
                {"eids": entity_ids}
            )
            triples = []
            while result.has_next():
                row = result.get_next()
                triples.append(Triple(
                    subject=row[0], subject_type=row[1], predicate=row[2],
                    object=row[3], object_type=row[4], source=row[5],
                    confidence=row[6], pmid=row[7] if row[7] else None
                ))
            return triples
        except:
            return []
    
    def get_stats(self) -> Dict[str, int]:
        stats = {"entities": 0, "documents": 0, "relations": 0, "mentions": 0}
        for key, query in [
            ("entities", "MATCH (e:Entity) RETURN count(e)"),
            ("documents", "MATCH (d:Document) RETURN count(d)"),
            ("relations", "MATCH ()-[r:RELATES_TO]->() RETURN count(r)"),
            ("mentions", "MATCH ()-[m:MENTIONS]->() RETURN count(m)")
        ]:
            try:
                result = self.conn.execute(query)
                if result.has_next():
                    stats[key] = result.get_next()[0]
            except:
                pass
        return stats
    
    def close(self):
        self.conn = None
        self.db = None
        self._initialized = False


# ============================================
# PRIMEKG LOADER
# ============================================

class PrimeKGLoader:
    """Load PrimeKG into Kuzu."""
    
    PRIMEKG_URL = "https://dataverse.harvard.edu/api/access/datafile/6180620"
    
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.edges_file = self.data_dir / "kg.csv"
        
    def download_if_needed(self) -> bool:
        if self.edges_file.exists():
            logger.info("PrimeKG CSV exists")
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
        """Load PrimeKG - SKIPS if already loaded."""
        stats = kg_db.get_stats()
        if stats.get('entities', 0) > 50000:
            logger.info(f"✓ PrimeKG already loaded: {stats['entities']:,} entities, {stats['relations']:,} relations")
            return stats.get('relations', 0)
        
        if not self.edges_file.exists():
            if not self.download_if_needed():
                return 0
        
        logger.info("Loading PrimeKG (5-15 min on first run)...")
        start_time = time.time()
        
        try:
            import pandas as pd
            df = pd.read_csv(self.edges_file, low_memory=False)
            if limit:
                df = df.head(limit)
            
            # Extract unique entities
            subjects = df[['x_id', 'x_type', 'x_name']].drop_duplicates()
            objects = df[['y_id', 'y_type', 'y_name']].drop_duplicates()
            all_entities = pd.concat([
                subjects.rename(columns={'x_id': 'id', 'x_type': 'type', 'x_name': 'name'}),
                objects.rename(columns={'y_id': 'id', 'y_type': 'type', 'y_name': 'name'})
            ]).drop_duplicates(subset=['id'])
            
            # Insert entities
            logger.info(f"Inserting {len(all_entities):,} entities...")
            entity_count = 0
            for _, row in all_entities.iterrows():
                kg_db.add_entity(f"primekg:{row['id']}", 
                                str(row['name']) if pd.notna(row['name']) else '',
                                str(row['type']) if pd.notna(row['type']) else '',
                                source="primekg")
                entity_count += 1
                if entity_count % 10000 == 0:
                    logger.info(f"  Entities: {entity_count:,}/{len(all_entities):,}")
            
            # Insert relations
            logger.info(f"Inserting {len(df):,} relations...")
            relation_count = 0
            for _, row in df.iterrows():
                kg_db.add_relation(f"primekg:{row['x_id']}", 
                                  str(row['relation']) if pd.notna(row['relation']) else '',
                                  f"primekg:{row['y_id']}", source="primekg")
                relation_count += 1
                if relation_count % 50000 == 0:
                    logger.info(f"  Relations: {relation_count:,}/{len(df):,}")
            
            logger.info(f"✓ Loaded in {time.time()-start_time:.1f}s: {entity_count:,} entities, {relation_count:,} relations")
            return relation_count
        except Exception as e:
            logger.error(f"Load failed: {e}")
            return 0


# ============================================
# ENTITY LINKER
# ============================================

class EntityLinker:
    """Link entities to KG nodes."""
    
    TYPE_MAP = {"Disease": "disease", "Drug": "drug", "Gene": "gene/protein", 
                "Protein": "gene/protein", "Chemical": "drug", "Pathway": "pathway"}
    
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
    
    def clear_cache(self):
        self.cache.clear()


# ============================================
# KG-ENHANCED RETRIEVER
# ============================================

class KGEnhancedRetriever:
    """Expand queries using KG."""
    
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
        
        return {
            'entities': entities,
            'linked_nodes': linked_nodes,
            'expanded_entities': expanded_entities,
            'relevant_triples': relevant_triples,
            'expanded_terms': list(set(expanded_terms))[:30]
        }
    
    def format_kg_context(self, triples: List[Triple], max_triples: int = 10) -> str:
        if not triples:
            return ""
        lines = ["[Knowledge Graph Context]"]
        for triple in triples[:max_triples]:
            lines.append(f"• {triple.subject} —[{triple.predicate}]→ {triple.object}")
        return "\n".join(lines)


# ============================================
# FACT VERIFIER
# ============================================

class FactVerifier:
    """Verify claims against KG."""
    
    RELATION_KEYWORDS = ['causes', 'treats', 'inhibits', 'activates', 'associated with',
                         'targets', 'binds to', 'regulates', 'linked to', 'prevents', 'induces']
    
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
        return {'total_claims': total, 'verified_claims': verified, 
                'verification_rate': verified / total if total > 0 else 1.0}


# ============================================
# KG DOCUMENT PROCESSOR
# ============================================

class KGDocumentProcessor:
    """Process papers and add to KG."""
    
    def __init__(self, kg_db: KuzuGraphDB, extractor: GLiNERExtractor, linker: EntityLinker):
        self.kg_db = kg_db
        self.extractor = extractor
        self.linker = linker
    
    def process_paper(self, pmid: str, title: str, abstract: str,
                      year: str, full_text: str = None) -> Dict[str, int]:
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
# KNOWLEDGE GRAPH MANAGER
# ============================================

class KnowledgeGraphManager:
    """Main interface for KG operations."""
    
    def __init__(self, db_path: Path, gliner_device: str = "cpu",
                 load_primekg: bool = True, primekg_limit: int = None,
                 primekg_data_dir: Path = None):
        
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
    
    def process_paper(self, pmid: str, title: str, abstract: str,
                      year: str, full_text: str = None) -> Dict:
        return self.doc_processor.process_paper(pmid, title, abstract, year, full_text)
    
    def find_entity(self, name: str, entity_type: str = None, limit: int = 10) -> List[Dict]:
        return self.kg_db.find_entities_by_name(name, entity_type, limit)
    
    def get_entity_neighbors(self, entity_id: str, limit: int = 50) -> List[Dict]:
        return self.kg_db.get_entity_neighbors(entity_id, limit=limit)
    
    def get_stats(self) -> Dict:
        return self.kg_db.get_stats()
    
    def unload_gliner(self):
        self.extractor.unload()
    
    def close(self):
        self.kg_db.close()


# Exports
__all__ = [
    'Entity', 'Triple', 'VerificationResult',
    'GLiNERExtractor', 'KuzuGraphDB', 'PrimeKGLoader', 'EntityLinker',
    'KGEnhancedRetriever', 'FactVerifier', 'KGDocumentProcessor', 'KnowledgeGraphManager'
]
