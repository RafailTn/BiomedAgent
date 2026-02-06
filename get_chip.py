"""
Epigenomic Data Query Tool for AI Agents
========================================
Unified interface to query histone modifications and TF binding peaks 
from ENCODE, ChIP-Atlas, UCSC, and DeepBlue.

Designed for AI agent consumption with structured, interpretable output.
"""

import requests
import json
import sys
from typing import Dict, List, Optional, Union, Literal
from dataclasses import dataclass, asdict
from urllib.parse import urlencode
import xmlrpc.client
from io import StringIO
import gzip


@dataclass
class CoordinateRange:
    """Genomic coordinate range"""
    chrom: str
    start: int
    end: int
    assembly: Literal["hg38", "GRCh38"] = "hg38"
    
    def __str__(self):
        return f"{self.chrom}:{self.start}-{self.end}"
    
    def to_bed(self) -> str:
        """Return in BED format"""
        return f"{self.chrom}\t{self.start}\t{self.end}"


@dataclass
class EpigenomicFeature:
    """Standardized epigenomic feature result"""
    source: str  # ENCODE, ChIP-Atlas, UCSC, DeepBlue
    experiment_id: str
    feature_type: str  # histone_modification, tf_binding, chromatin_accessibility
    target: str  # H3K27ac, CTCF, etc.
    biosample: str
    coordinates: CoordinateRange
    score: Optional[float] = None
    signal_value: Optional[float] = None
    q_value: Optional[float] = None
    p_value: Optional[float] = None
    peak_calling_software: Optional[str] = None
    file_format: str = "bed"
    download_url: Optional[str] = None
    metadata: Dict = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_ai_summary(self) -> str:
        """Generate human-readable summary for AI interpretation"""
        confidence = "high" if (self.q_value and self.q_value < 1e-10) else "moderate" if (self.q_value and self.q_value < 1e-5) else "low"
        return (f"[{self.source}] {self.target} binding/modification detected in {self.biosample} "
                f"at {self.coordinates} (confidence: {confidence}, score: {self.score:.2f} if self.score else 'N/A')")


class EpigenomicQueryTool:
    """
    Unified tool for querying epigenomic data from multiple sources.
    Optimized for fast coordinate-range queries across tissues/cell types.
    """
    
    def __init__(self):
        self.encode_base = "https://www.encodeproject.org"
        self.chip_atlas_base = "https://chip-atlas.dbcls.jp"
        self.ucsc_api = "https://api.genome.ucsc.edu"
        self.deepblue_url = "http://deepblue.mpi-inf.mpg.de/xmlrpc"
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
    
    # ==================== ENCODE API Methods ====================
    
    def query_encode(
        self, 
        coords: CoordinateRange,
        assay_type: Literal["ChIP-seq", "DNase-seq", "ATAC-seq"] = "ChIP-seq",
        target: Optional[str] = None,  # H3K27ac, CTCF, etc.
        biosample: Optional[str] = None,
        file_format: str = "bed"
    ) -> List[EpigenomicFeature]:
        """
        Query ENCODE for experiments and files overlapping coordinates.
        Returns file metadata and download URLs for manual retrieval.
        """
        results = []
        
        # Build search query for experiments
        search_params = {
            "type": "Experiment",
            "assay_title": assay_type,
            "status": "released",
            "assembly": coords.assembly if coords.assembly in ["hg19", "mm9", "mm10"] else "GRCh38",
            "format": "json",
            "frame": "embedded",
            "limit": "all"
        }
        
        if target:
            search_params["target.label"] = target
        
        if biosample:
            search_params["biosample_ontology.term_name"] = biosample
        
        # Search experiments
        search_url = f"{self.encode_base}/search/?{urlencode(search_params)}"
        response = self.session.get(search_url)
        response.raise_for_status()
        data = response.json()
        
        # Process results
        for exp in data.get("@graph", []):
            exp_id = exp.get("accession", "unknown")
            exp_biosample = exp.get("biosample_ontology", {}).get("term_name", "unknown")
            exp_target = exp.get("target", {}).get("label", "unknown") if exp.get("target") else "input"
            
            # Get files for this experiment
            for file_obj in exp.get("files", []):
                if file_obj.get("file_format") != file_format:
                    continue
                
                file_assembly = file_obj.get("assembly", "unknown")
                if file_assembly != coords.assembly and not (
                    (file_assembly == "GRCh38" and coords.assembly == "hg38") or
                    (file_assembly == "hg38" and coords.assembly == "GRCh38")
                ):
                    continue
                
                # Check if file overlaps coordinates (simplified check)
                # ENCODE files don't provide coordinates in metadata, so we return all matching files
                # The actual overlap check would require downloading and parsing the file
                
                feature = EpigenomicFeature(
                    source="ENCODE",
                    experiment_id=exp_id,
                    feature_type="histone_modification" if "H3" in str(exp_target) else "tf_binding" if exp_target != "input" else "input_control",
                    target=exp_target,
                    biosample=exp_biosample,
                    coordinates=coords,
                    file_format=file_format,
                    download_url=f"{self.encode_base}{file_obj.get('href')}" if file_obj.get('href') else None,
                    metadata={
                        "file_accession": file_obj.get("accession"),
                        "output_type": file_obj.get("output_type"),
                        "biological_replicates": file_obj.get("biological_replicates", []),
                        "technical_replicates": file_obj.get("technical_replicates", [])
                    }
                )
                results.append(feature)
        
        return results
    
    def get_encode_file_data(self, file_url: str, coords: CoordinateRange) -> List[Dict]:
        """
        Download and parse ENCODE BED file, returning only regions overlapping coords.
        Note: This downloads the full file which can be large. Consider using UCSC API for 
        coordinate-specific queries instead.
        """
        try:
            response = self.session.get(file_url, stream=True)
            response.raise_for_status()
            
            # Handle gzipped files
            if file_url.endswith('.gz'):
                content = gzip.decompress(response.content).decode('utf-8')
            else:
                content = response.content.decode('utf-8')
            
            overlapping = []
            for line in content.strip().split('\n'):
                if line.startswith('#') or line.startswith('track') or line.startswith('browser'):
                    continue
                cols = line.split('\t')
                if len(cols) >= 3:
                    chrom, start, end = cols[0], int(cols[1]), int(cols[2])
                    if chrom == coords.chrom and not (end < coords.start or start > coords.end):
                        entry = {
                            "chrom": chrom,
                            "start": start,
                            "end": end,
                            "name": cols[3] if len(cols) > 3 else None,
                            "score": float(cols[4]) if len(cols) > 4 and cols[4].replace('.','').isdigit() else None,
                            "strand": cols[5] if len(cols) > 5 else None,
                            "signal_value": float(cols[6]) if len(cols) > 6 and cols[6].replace('.','').isdigit() else None,
                            "p_value": float(cols[7]) if len(cols) > 7 and cols[7].replace('.','').isdigit() else None,
                            "q_value": float(cols[8]) if len(cols) > 8 and cols[8].replace('.','').isdigit() else None
                        }
                        overlapping.append(entry)
            return overlapping
        except Exception as e:
            return [{"error": str(e), "file_url": file_url}]
    
    # ==================== ChIP-Atlas API Methods ====================
    
    def query_chip_atlas(
        self,
        coords: CoordinateRange,
        antigen: Optional[str] = None,  # CTCF, H3K27ac, etc.
        cell_type: Optional[str] = None,  # blood, liver, etc.
        threshold: Literal["05", "10", "20"] = "10"
    ) -> List[EpigenomicFeature]:
        """
        Query ChIP-Atlas for peaks in coordinate range.
        ChIP-Atlas provides pre-processed peak calls from SRA data.
        """
        results = []
        
        # ChIP-Atlas uses a different coordinate format for their API
        # They provide bulk downloads, but for specific queries we use their search interface
        # to get experiment IDs, then construct download URLs
        
        # First, search for experiments
        search_params = {
            "genome": self._convert_assembly_chipatlas(coords.assembly),
            "antigen": antigen or "",
            "celltype": cell_type or "",
            "format": "json"
        }
        
        # ChIP-Atlas experiment search
        search_url = f"{self.chip_atlas_base}/api/search"
        try:
            response = self.session.get(search_url, params=search_params, timeout=30)
            if response.status_code == 200:
                experiments = response.json()
                
                for exp in experiments.get("data", []):
                    exp_id = exp.get("srx", "unknown")
                    exp_antigen = exp.get("antigen", "unknown")
                    exp_cell = exp.get("celltype", "unknown")
                    
                    # Construct direct download URL for peak file
                    # ChIP-Atlas stores peaks in bed/bigBed format accessible via pattern
                    download_url = self._construct_chip_atlas_url(
                        coords.assembly, exp_id, threshold
                    )
                    
                    feature = EpigenomicFeature(
                        source="ChIP-Atlas",
                        experiment_id=exp_id,
                        feature_type="histone_modification" if "H3" in exp_antigen else "tf_binding",
                        target=exp_antigen,
                        biosample=exp_cell,
                        coordinates=coords,
                        file_format="bed",
                        download_url=download_url,
                        peak_calling_software="MACS2",
                        metadata={
                            "pmid": exp.get("pmid"),
                            "title": exp.get("title"),
                            "threshold": threshold
                        }
                    )
                    results.append(feature)
        except requests.RequestException as e:
            results.append(EpigenomicFeature(
                source="ChIP-Atlas",
                experiment_id="error",
                feature_type="error",
                target=str(e),
                biosample="",
                coordinates=coords
            ))
        
        return results
    
    def _convert_assembly_chipatlas(self, assembly: str) -> str:
        """Convert assembly names to ChIP-Atlas format"""
        mapping = {
            "hg38": "hg38",
            "GRCh38": "hg38",
            "hg19": "hg19",
            "mm10": "mm10",
            "mm9": "mm9"
        }
        return mapping.get(assembly, "hg38")
    
    def _construct_chip_atlas_url(self, assembly: str, srx: str, threshold: str) -> str:
        """Construct ChIP-Atlas peak file download URL"""
        asm = self._convert_assembly_chipatlas(assembly)
        # ChIP-Atlas stores files in a specific pattern
        base = f"http://dbarchive.biosciencedbc.jp/kyushu-u/{asm}/eachData/bed{threshold}"
        return f"{base}/{srx}.{threshold}.bed"
    
    # ==================== UCSC API Methods ====================
    
    def query_ucsc_track(
        self,
        coords: CoordinateRange,
        track: str = "wgEncodeRegTfbsClusteredV3",  # ENCODE TFBS clusters
        hub_url: Optional[str] = None
    ) -> List[EpigenomicFeature]:
        """
        Query UCSC Genome Browser API for track data in specific coordinates.
        Fastest method for coordinate-specific queries - returns data directly.
        """
        results = []
        
        params = {
            "genome": coords.assembly,
            "chrom": coords.chrom,
            "start": coords.start,
            "end": coords.end,
            "track": track
        }
        
        if hub_url:
            params["hubUrl"] = hub_url
        
        url = f"{self.ucsc_api}/getData/track"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Parse UCSC track data format
            track_data = data.get(track, [])
            
            for item in track_data:
                # Handle different UCSC track formats
                chrom = item.get("chrom", coords.chrom)
                start = item.get("chromStart", item.get("start", 0))
                end = item.get("chromEnd", item.get("end", 0))
                
                feature = EpigenomicFeature(
                    source="UCSC",
                    experiment_id=f"{track}_{chrom}_{start}",
                    feature_type="tf_binding" if "Tfbs" in track else "histone_modification" if "Histone" in track else "regulatory",
                    target=item.get("name", track),
                    biosample=item.get("cellType", item.get("tissue", "unknown")),
                    coordinates=CoordinateRange(chrom, start, end, coords.assembly),
                    score=item.get("score"),
                    signal_value=item.get("signalValue"),
                    p_value=item.get("pValue"),
                    q_value=item.get("qValue"),
                    metadata=item
                )
                results.append(feature)
                
        except requests.RequestException as e:
            results.append(EpigenomicFeature(
                source="UCSC",
                experiment_id="error",
                feature_type="error",
                target=str(e),
                biosample="",
                coordinates=coords
            ))
        
        return results
    
    def list_ucsc_tracks(self, assembly: str = "hg38", hub_url: Optional[str] = None) -> List[Dict]:
        """List available tracks for an assembly"""
        params = {"genome": assembly}
        if hub_url:
            params["hubUrl"] = hub_url
        
        url = f"{self.ucsc_api}/list/tracks"
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()
    
    # ==================== DeepBlue API Methods ====================
    
    def query_deepblue(
        self,
        coords: CoordinateRange,
        epigenetic_mark: Optional[str] = None,
        biosource: Optional[str] = None,
        technique: Optional[str] = "ChIP-seq"
    ) -> List[EpigenomicFeature]:
        """
        Query DeepBlue epigenomic data server.
        Best for aggregating data across multiple experiments.
        """
        results = []
        
        try:
            # DeepBlue uses XML-RPC
            server = xmlrpc.client.ServerProxy(self.deepblue_url, allow_none=True)
            
            # Search for experiments
            query = {
                "genome": coords.assembly,
                "technique": technique
            }
            if epigenetic_mark:
                query["epigenetic_mark"] = epigenetic_mark
            if biosource:
                query["biosource"] = biosource
            
            # List experiments matching criteria
            status, exp_ids = server.list_experiments(query, "true", "true", None)
            
            if status == "okay":
                for exp_id in exp_ids:
                    # Get experiment info
                    status, info = server.info(exp_id, None)
                    if status == "okay":
                        info = info[0]
                        
                        # Select regions in coordinate range
                        status, query_id = server.select_regions(
                            exp_id, 
                            coords.chrom, 
                            coords.start, 
                            coords.end, 
                            None
                        )
                        
                        if status == "okay":
                            # Get regions data
                            status, regions = server.get_regions(query_id, "CHROMOSOME,START,END,NAME,SCORE,STRAND,SIGNAL_VALUE,P_VALUE,Q_VALUE", None)
                            
                            if status == "okay" and regions:
                                for region in regions.split('\n'):
                                    if region.strip():
                                        cols = region.split('\t')
                                        if len(cols) >= 3:
                                            feature = EpigenomicFeature(
                                                source="DeepBlue",
                                                experiment_id=exp_id,
                                                feature_type="histone_modification" if "H3" in str(info.get("epigenetic_mark")) else "tf_binding",
                                                target=info.get("epigenetic_mark", "unknown"),
                                                biosource=info.get("biosource", "unknown"),
                                                coordinates=CoordinateRange(
                                                    cols[0], 
                                                    int(cols[1]), 
                                                    int(cols[2]), 
                                                    coords.assembly
                                                ),
                                                name=cols[3] if len(cols) > 3 else None,
                                                score=float(cols[4]) if len(cols) > 4 and cols[4] else None,
                                                signal_value=float(cols[6]) if len(cols) > 6 and cols[6] else None,
                                                p_value=float(cols[7]) if len(cols) > 7 and cols[7] else None,
                                                q_value=float(cols[8]) if len(cols) > 8 and cols[8] else None,
                                                metadata=info
                                            )
                                            results.append(feature)
        except Exception as e:
            results.append(EpigenomicFeature(
                source="DeepBlue",
                experiment_id="error",
                feature_type="error",
                target=str(e),
                biosample="",
                coordinates=coords
            ))
        
        return results
    
    # ==================== Unified Query Interface ====================
    
    def query_all(
        self,
        chrom: str,
        start: int,
        end: int,
        assembly: Literal["hg19", "hg38", "mm9", "mm10", "GRCh38"] = "hg38",
        target: Optional[str] = None,
        biosample: Optional[str] = None,
        sources: List[Literal["encode", "chip_atlas", "ucsc", "deepblue"]] = None
    ) -> Dict:
        """
        Query all available sources for epigenomic features in coordinate range.
        Returns structured results optimized for AI agent consumption.
        """
        if sources is None:
            sources = ["encode", "chip_atlas", "ucsc"]
        
        coords = CoordinateRange(chrom, start, end, assembly)
        all_results = {
            "query": {
                "coordinates": str(coords),
                "assembly": assembly,
                "target": target,
                "biosample": biosample,
                "sources_queried": sources
            },
            "results": [],
            "summary": {},
            "errors": []
        }
        
        # Query each source
        for source in sources:
            try:
                if source == "encode":
                    features = self.query_encode(coords, target=target, biosample=biosample)
                elif source == "chip_atlas":
                    features = self.query_chip_atlas(coords, antigen=target, cell_type=biosample)
                elif source == "ucsc":
                    # Use appropriate track based on target
                    track = self._select_ucsc_track(target)
                    features = self.query_ucsc_track(coords, track=track)
                elif source == "deepblue":
                    features = self.query_deepblue(coords, epigenetic_mark=target, biosource=biosample)
                else:
                    continue
                
                all_results["results"].extend([f.to_dict() for f in features])
                
            except Exception as e:
                all_results["errors"].append({"source": source, "error": str(e)})
        
        # Generate summary statistics
        all_results["summary"] = self._generate_summary(all_results["results"])
        
        return all_results
    
    def _select_ucsc_track(self, target: Optional[str]) -> str:
        """Select appropriate UCSC track based on target"""
        if target is None:
            return "wgEncodeRegTfbsClusteredV3"  # ENCODE TFBS clusters
        
        target_upper = target.upper()
        if "H3" in target_upper:
            # Histone modification tracks
            return "wgEncodeBroadHistone"
        elif target_upper in ["CTCF", "POLR2A", "EP300", "RAD21"]:
            return "wgEncodeRegTfbsClusteredV3"
        else:
            return "wgEncodeRegTfbsClusteredV3"
    
    def _generate_summary(self, results: List[Dict]) -> Dict:
        """Generate summary statistics for AI interpretation"""
        if not results:
            return {"total_features": 0, "sources": {}, "targets": {}, "biosamples": {}}
        
        summary = {
            "total_features": len(results),
            "sources": {},
            "feature_types": {},
            "targets": {},
            "biosamples": {}
        }
        
        for r in results:
            # Count by source
            src = r.get("source", "unknown")
            summary["sources"][src] = summary["sources"].get(src, 0) + 1
            
            # Count by feature type
            ft = r.get("feature_type", "unknown")
            summary["feature_types"][ft] = summary["feature_types"].get(ft, 0) + 1
            
            # Count by target
            tgt = r.get("target", "unknown")
            summary["targets"][tgt] = summary["targets"].get(tgt, 0) + 1
            
            # Count by biosample
            bio = r.get("biosample", "unknown")
            summary["biosamples"][bio] = summary["biosamples"].get(bio, 0) + 1
        
        return summary
    
    def get_ai_interpretation_prompt(self, results: Dict) -> str:
        """
        Generate a prompt-ready summary for AI agent consumption.
        This formats the results in a way that's easy for LLMs to interpret.
        """
        if not results["results"]:
            return f"No epigenomic features found for {results['query']['coordinates']}."
        
        lines = [
            f"Epigenomic Analysis Results for {results['query']['coordinates']}",
            f"Assembly: {results['query']['assembly']}",
            f"Query Target: {results['query']['target'] or 'Any'}",
            f"Biosample: {results['query']['biosample'] or 'Any'}",
            "",
            f"Total Features Found: {results['summary']['total_features']}",
            "",
            "Breakdown by Source:",
        ]
        
        for source, count in results["summary"]["sources"].items():
            lines.append(f"  - {source}: {count} features")
        
        lines.extend(["", "Breakdown by Target:"])
        for target, count in results["summary"]["targets"].items():
            lines.append(f"  - {target}: {count} occurrences")
        
        lines.extend(["", "Breakdown by Biosample:"])
        for bio, count in list(results["summary"]["biosamples"].items())[:10]:  # Top 10
            lines.append(f"  - {bio}: {count} features")
        
        lines.extend(["", "Key Findings:"])
        
        # Identify high-confidence peaks
        high_conf = [r for r in results["results"] if r.get("q_value") and r["q_value"] < 1e-10]
        if high_conf:
            lines.append(f"- Found {len(high_conf)} high-confidence peaks (q < 1e-10)")
            unique_targets = set(r["target"] for r in high_conf)
            lines.append(f"- High-confidence targets: {', '.join(unique_targets)}")
        
        # Identify tissue-specific patterns
        if len(results["summary"]["biosamples"]) > 1:
            lines.append(f"- Data available across {len(results['summary']['biosamples'])} different biosamples")
        
        return "\n".join(lines)


# ==================== CLI Interface for Direct Usage ====================

def main():
    """Command-line interface for the epigenomic query tool"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Query epigenomic data (histone mods, TF binding) from ENCODE, ChIP-Atlas, UCSC"
    )
    parser.add_argument("chrom", help="Chromosome (e.g., chr1)")
    parser.add_argument("start", type=int, help="Start coordinate")
    parser.add_argument("end", type=int, help="End coordinate")
    parser.add_argument("--assembly", default="hg38", choices=["hg19", "hg38", "mm9", "mm10", "GRCh38"])
    parser.add_argument("--target", "-t", help="Target protein or histone mark (e.g., CTCF, H3K27ac)")
    parser.add_argument("--biosample", "-b", help="Biosample/cell type (e.g., K562, liver)")
    parser.add_argument("--sources", "-s", nargs="+", default=["encode", "ucsc"],
                       choices=["encode", "chip_atlas", "ucsc", "deepblue"],
                       help="Data sources to query")
    parser.add_argument("--format", "-f", choices=["json", "ai", "bed"], default="json",
                       help="Output format")
    
    args = parser.parse_args()
    
    tool = EpigenomicQueryTool()
    
    results = tool.query_all(
        chrom=args.chrom,
        start=args.start,
        end=args.end,
        assembly=args.assembly,
        target=args.target,
        biosample=args.biosample,
        sources=args.sources
    )
    
    if args.format == "json":
        print(json.dumps(results, indent=2))
    elif args.format == "ai":
        print(tool.get_ai_interpretation_prompt(results))
    elif args.format == "bed":
        # Output in BED format
        for r in results["results"]:
            if "coordinates" in r:
                c = r["coordinates"]
                print(f"{c['chrom']}\t{c['start']}\t{c['end']}\t{r['source']}:{r['target']}\t{r.get('score', 0)}")


if __name__ == "__main__":
    main()
