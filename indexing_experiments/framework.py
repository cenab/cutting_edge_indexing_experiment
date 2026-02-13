from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer

try:
    import networkx as nx
except Exception:
    nx = None

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None


TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")
ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b")
SCENE_RE = re.compile(r"\[SCENE\s*:\s*([^\]]+)\]", re.IGNORECASE)


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    category: str
    name: str
    description: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


METHOD_SPECS: List[MethodSpec] = [
    MethodSpec("1.1_fixed_length_chunking", "Chunk-Based Indexing", "Fixed-Length Chunking", "Uniform token windows with predictable memory usage."),
    MethodSpec("1.2_sliding_window_overlap", "Chunk-Based Indexing", "Sliding Window with Overlap", "Overlapping windows to preserve context transitions."),
    MethodSpec("1.3_adaptive_semantic_chunking", "Chunk-Based Indexing", "Adaptive Semantic Chunking", "Chunks generated using semantic coherence boundaries."),
    MethodSpec("1.4_query_aware_dynamic_chunking", "Chunk-Based Indexing", "Query-Aware Dynamic Chunking", "Query-time chunk assembly around intent-bearing spans."),
    MethodSpec("1.5_small_to_big_hierarchical_chunking", "Chunk-Based Indexing", "Small-to-Big Hierarchical Chunking", "Retrieve fine chunks and respond with parent context."),
    MethodSpec("1.6_sentence_level_indexing", "Chunk-Based Indexing", "Sentence-Level Indexing", "Sentence-granular retrieval with local context links."),
    MethodSpec("2.1_flat_dense_index", "Vector-Based Indexing", "Flat Dense Index", "Exhaustive dense retrieval across all embeddings."),
    MethodSpec("2.2_approximate_nearest_neighbor_index", "Vector-Based Indexing", "Approximate Nearest Neighbor Index", "Cluster-pruned candidate search for faster retrieval."),
    MethodSpec("2.3_quantized_vector_index", "Vector-Based Indexing", "Quantized Vector Index", "Compressed vectors for reduced memory footprint."),
    MethodSpec("2.4_hybrid_dense_partition_index", "Vector-Based Indexing", "Hybrid Dense Partition Index", "Partition routing followed by dense rerank."),
    MethodSpec("3.1_inverted_index", "Sparse and Lexical Indexing", "Inverted Index", "Term-to-postings lexical retrieval."),
    MethodSpec("3.2_sparse_dense_fusion_index", "Sparse and Lexical Indexing", "Sparse-Dense Fusion Index", "Weighted fusion of lexical and dense scores."),
    MethodSpec("3.3_weighted_term_index", "Sparse and Lexical Indexing", "Weighted Term Index", "BM25-style weighted term relevance ranking."),
    MethodSpec("4.1_tree_structured_index", "Hierarchical Indexing", "Tree-Structured Index", "Parent-child index with section-aware retrieval."),
    MethodSpec("4.2_summary_augmented_index", "Hierarchical Indexing", "Summary-Augmented Index", "Summary-first routing to full chunks."),
    MethodSpec("4.3_structural_aware_index", "Hierarchical Indexing", "Structural-Aware Index", "Chunking aligned with headings and structural blocks."),
    MethodSpec("5.1_knowledge_graph_index", "Graph-Based Indexing", "Knowledge Graph Index", "Entity-passage graph with relation-aware retrieval."),
    MethodSpec("5.2_entity_centric_index", "Graph-Based Indexing", "Entity-Centric Index", "Entity-first retrieval and passage linkage."),
    MethodSpec("5.3_semantic_similarity_graph", "Graph-Based Indexing", "Semantic Similarity Graph", "kNN graph over semantic embeddings."),
    MethodSpec("5.4_multi_hop_retrieval_graph", "Graph-Based Indexing", "Multi-Hop Retrieval Graph", "Graph traversal for multi-hop evidence gathering."),
    MethodSpec("6.1_topic_partition_index", "Topic and Cluster-Based Indexing", "Topic Partition Index", "Query routing into thematic clusters."),
    MethodSpec("6.2_semantic_hashing_index", "Topic and Cluster-Based Indexing", "Semantic Hashing Index", "Binary hash retrieval with hamming filtering."),
    MethodSpec("7.1_time_decayed_index", "Temporal and Streaming Indexing", "Time-Decayed Index", "Recency-aware ranking with temporal decay."),
    MethodSpec("7.2_sliding_temporal_window_index", "Temporal and Streaming Indexing", "Sliding Temporal Window Index", "Separate recent and historical retrieval windows."),
    MethodSpec("7.3_real_time_streaming_index", "Temporal and Streaming Indexing", "Real-Time Streaming Index", "Incremental insertion without full rebuild."),
    MethodSpec("8.1_trust_weighted_index", "Robust and Adversarial-Aware Indexing", "Trust-Weighted Index", "Provenance-weighted ranking."),
    MethodSpec("8.2_outlier_aware_vector_index", "Robust and Adversarial-Aware Indexing", "Outlier-Aware Vector Index", "Outlier filtering at insertion/search."),
    MethodSpec("8.3_drift_aware_index", "Robust and Adversarial-Aware Indexing", "Drift-Aware Index", "Distribution-shift-aware ranking controls."),
    MethodSpec("9.1_distributional_embedding_index", "Probabilistic and Uncertainty-Aware Indexing", "Distributional Embedding Index", "Mean/variance vector scoring."),
    MethodSpec("9.2_confidence_propagating_index", "Probabilistic and Uncertainty-Aware Indexing", "Confidence-Propagating Index", "Uncertainty-adjusted confidence ranking."),
    MethodSpec("10.1_joint_embedding_index", "Multi-Modal Indexing", "Joint Embedding Index", "Shared embedding space for mixed modalities."),
    MethodSpec("10.2_cross_modal_linked_index", "Multi-Modal Indexing", "Cross-Modal Linked Index", "Modality-specific nodes linked by alignment edges."),
    MethodSpec("10.3_scene_aware_media_index", "Multi-Modal Indexing", "Scene-Aware Media Index", "Scene segmentation with OCR/transcript metadata."),
    MethodSpec("11.1_self_tuning_index", "Adaptive and Continual Indexing", "Self-Tuning Index", "Automatic parameter adaptation to corpus characteristics."),
    MethodSpec("11.2_feedback_driven_index", "Adaptive and Continual Indexing", "Feedback-Driven Index", "Ranking updates from interaction feedback."),
    MethodSpec("11.3_continual_re_embedding_index", "Adaptive and Continual Indexing", "Continual Re-Embedding Index", "Periodic re-embedding for model upgrades."),
    MethodSpec("12.1_causal_graph_index", "Reasoning-Ready Indexing", "Causal Graph Index", "Cause-effect relation graph retrieval."),
    MethodSpec("12.2_explanation_aware_index", "Reasoning-Ready Indexing", "Explanation-Aware Index", "Reasoning chain annotations in ranking."),
    MethodSpec("12.3_plan_oriented_index", "Reasoning-Ready Indexing", "Plan-Oriented Index", "Action/procedure-aware content annotation."),
]

METHOD_SPEC_MAP: Dict[str, MethodSpec] = {m.method_id: m for m in METHOD_SPECS}


def method_slug(method_id: str) -> str:
    return method_id.replace(".", "_")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in pieces if p.strip()]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.strptime(value[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return scores
    valid = np.isfinite(scores)
    if not valid.any():
        return np.zeros_like(scores)
    v = scores.copy()
    min_v = np.min(v[valid])
    max_v = np.max(v[valid])
    if math.isclose(min_v, max_v):
        out = np.zeros_like(v)
        out[valid] = 1.0
        return out
    out = np.zeros_like(v)
    out[valid] = (v[valid] - min_v) / (max_v - min_v)
    return out


class DenseEmbedder:
    def __init__(self, n_components: int = 128, ngram_range: Tuple[int, int] = (1, 2)) -> None:
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=1)
        self.reducer: Optional[TruncatedSVD] = None
        self.normalizer = Normalizer(copy=False)

    def fit(self, texts: Sequence[str]) -> np.ndarray:
        x = self.vectorizer.fit_transform(texts)
        dense = x.toarray()
        if dense.shape[0] > 2 and dense.shape[1] > 2:
            max_components = min(self.n_components, dense.shape[0] - 1, dense.shape[1] - 1)
            if max_components >= 2:
                self.reducer = TruncatedSVD(n_components=max_components, random_state=42)
                dense = self.reducer.fit_transform(x)
        dense = self.normalizer.fit_transform(dense)
        return dense.astype(np.float32)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        x = self.vectorizer.transform(texts)
        if self.reducer is not None:
            dense = self.reducer.transform(x)
        else:
            dense = x.toarray()
        dense = self.normalizer.transform(dense)
        return dense.astype(np.float32)


def chunk_fixed(text: str, chunk_size: int = 120) -> List[str]:
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]


def chunk_sliding(text: str, chunk_size: int = 120, overlap: int = 40) -> List[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, chunk_size - overlap)
    chunks = []
    for i in range(0, len(words), step):
        part = words[i : i + chunk_size]
        if not part:
            continue
        chunks.append(" ".join(part))
        if i + chunk_size >= len(words):
            break
    return chunks


def chunk_sentences(text: str) -> List[str]:
    return split_sentences(text)


def chunk_semantic_adaptive(text: str, threshold: float = 0.2) -> List[str]:
    sentences = split_sentences(text)
    if len(sentences) <= 2:
        return sentences
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    x = vectorizer.fit_transform(sentences).toarray().astype(np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    x = x / norms

    chunks: List[List[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(np.dot(x[i - 1], x[i]))
        if sim < threshold and len(chunks[-1]) >= 2:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])
    return [" ".join(c) for c in chunks if c]


def chunk_structural(text: str) -> List[Tuple[str, int, str]]:
    lines = text.splitlines() or [text]
    heading = "root"
    depth = 0
    buf: List[str] = []
    out: List[Tuple[str, int, str]] = []

    def flush() -> None:
        if buf:
            block = "\n".join(buf).strip()
            if block:
                out.append((heading, depth, block))

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            flush()
            buf = []
            hashes = len(stripped) - len(stripped.lstrip("#"))
            heading = stripped[hashes:].strip() or "section"
            depth = hashes
        else:
            buf.append(line)
    flush()
    if not out:
        out.append(("root", 0, text))
    return out


def lexical_diversity(text: str) -> float:
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def extract_entities(text: str) -> List[str]:
    entities = [e.strip() for e in ENTITY_RE.findall(text)]
    return [e for e in entities if len(e) > 2]


def extract_scene_chunks(text: str) -> List[Tuple[str, str]]:
    matches = list(SCENE_RE.finditer(text))
    if not matches:
        return [("full", text)]
    chunks: List[Tuple[str, str]] = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        scene_name = match.group(1).strip().lower().replace(" ", "_")
        body = text[start:end].strip()
        if body:
            chunks.append((scene_name, body))
    return chunks or [("full", text)]


def default_corpus_records() -> List[Dict[str, Any]]:
    return [
        {
            "id": "doc_001",
            "title": "Industrial Revolution Grade 10",
            "text": "# Industrial Revolution\n## Grade 10 Overview\nThe Industrial Revolution began in Britain and spread through Europe. Factories expanded textile production and urban migration increased. Steam engines improved transport and manufacturing.",
            "timestamp": "2025-01-12",
            "trust_score": 0.90,
            "modality": "text",
            "asset_id": "asset_ir_text",
            "grade_level": "grade_10",
        },
        {
            "id": "doc_002",
            "title": "Industrial Revolution Grade 12",
            "text": "# Industrial Revolution\n## Grade 12 Analysis\nAt Grade 12 depth, analysis includes labor exploitation, global supply chains, and colonial extraction. Mechanization increased productivity but widened inequality. This perspective contrasts with introductory summaries.",
            "timestamp": "2025-01-15",
            "trust_score": 0.92,
            "modality": "text",
            "asset_id": "asset_ir_text",
            "grade_level": "grade_12",
        },
        {
            "id": "doc_003",
            "title": "French Revolution Causes",
            "text": "# French Revolution\nThe Revolution occurred because fiscal crisis, social inequality, and political distrust converged. Bread prices rose, tax burdens were uneven, and institutional legitimacy collapsed.",
            "timestamp": "2024-11-03",
            "trust_score": 0.95,
            "modality": "text",
            "asset_id": "asset_fr",
        },
        {
            "id": "doc_004",
            "title": "World War I Timeline",
            "text": "# World War I\n## Timeline\n1914 mobilization accelerated alliances. 1916 attrition battles reshaped strategy. 1918 armistice ended frontline fighting.",
            "timestamp": "2024-09-01",
            "trust_score": 0.88,
            "modality": "text",
            "asset_id": "asset_ww1",
        },
        {
            "id": "doc_005",
            "title": "Civil Rights Movement",
            "text": "# Civil Rights Movement\nThe movement is significant because coordinated legal strategy and mass mobilization challenged segregation. Therefore, federal legislation shifted enforcement and voting protections.",
            "timestamp": "2025-02-10",
            "trust_score": 0.93,
            "modality": "text",
            "asset_id": "asset_cr",
        },
        {
            "id": "doc_006",
            "title": "Renewable Energy Policy 2025",
            "text": "# Energy Update\nRecent policy expanded grid storage incentives and offshore wind permitting. The newest update in 2025 emphasizes rapid interconnection approvals.",
            "timestamp": "2025-12-02",
            "trust_score": 0.89,
            "modality": "text",
            "asset_id": "asset_energy",
        },
        {
            "id": "doc_007",
            "title": "Cybersecurity Rumor Thread",
            "text": "Unverified post claims all banks were breached overnight. No evidence is provided and details conflict across reposts.",
            "timestamp": "2025-12-10",
            "trust_score": 0.25,
            "modality": "text",
            "asset_id": "asset_cyber",
        },
        {
            "id": "doc_008",
            "title": "Cybersecurity Incident Bulletin",
            "text": "# Incident Bulletin\nA regional outage occurred due to a misconfigured firewall policy. Forensics found no confirmed data exfiltration. Recommended controls include staged rollout and immutable logging.",
            "timestamp": "2025-12-11",
            "trust_score": 0.97,
            "modality": "text",
            "asset_id": "asset_cyber",
        },
        {
            "id": "doc_009",
            "title": "Steam Engine Diagram OCR",
            "text": "[OCR] Watt condenser reduces energy loss. Boiler pressure must be monitored for safety.",
            "timestamp": "2025-03-08",
            "trust_score": 0.86,
            "modality": "image",
            "asset_id": "asset_steam_1",
        },
        {
            "id": "doc_010",
            "title": "Steam Engine Lecture Audio",
            "text": "[TRANSCRIPT] Audio lecture explains that steam engines convert thermal energy into mechanical work and transformed factory throughput.",
            "timestamp": "2025-03-08",
            "trust_score": 0.88,
            "modality": "audio",
            "asset_id": "asset_steam_1",
        },
        {
            "id": "doc_011",
            "title": "Factory Conditions Documentary",
            "text": "[SCENE: opening] Narrator introduces urban growth. [SCENE: factory_floor] Workers describe long shifts and ventilation hazards. [SCENE: reform] Later labor laws reduce child labor.",
            "timestamp": "2025-04-01",
            "trust_score": 0.90,
            "modality": "video",
            "asset_id": "asset_factory_video",
        },
        {
            "id": "doc_012",
            "title": "Archiving Procedure",
            "text": "Step 1: Inventory materials and label provenance. Step 2: Scan with checksum validation. Step 3: Store originals in humidity-controlled cabinets. Then publish a searchable catalog.",
            "timestamp": "2025-07-21",
            "trust_score": 0.94,
            "modality": "text",
            "asset_id": "asset_archive",
        },
        {
            "id": "doc_013",
            "title": "Climate Transition Brief",
            "text": "# Climate Brief\nGrid modernization and heat-pump adoption accelerated in late 2025. Financing barriers remain but deployment rates improved compared with 2024.",
            "timestamp": "2025-11-19",
            "trust_score": 0.91,
            "modality": "text",
            "asset_id": "asset_energy",
        },
        {
            "id": "doc_014",
            "title": "Adversarial Keyword Stuffing",
            "text": "industrial revolution industrial revolution industrial revolution miracle cure click now now now. unrelated payload and random symbols.",
            "timestamp": "2025-10-10",
            "trust_score": 0.10,
            "modality": "text",
            "asset_id": "asset_noise",
        },
        {
            "id": "doc_015",
            "title": "Supply Chain Delay Analysis",
            "text": "Delays happened because port congestion and component shortages led to cascading schedule slips. Therefore procurement teams shifted to dual sourcing.",
            "timestamp": "2025-08-14",
            "trust_score": 0.87,
            "modality": "text",
            "asset_id": "asset_supply",
        },
        {
            "id": "doc_016",
            "title": "Explanation Skills Guide",
            "text": "To explain a concept clearly, start with prior knowledge, then connect new evidence, and finally verify understanding. This means explanations should include reasoned transitions.",
            "timestamp": "2025-05-29",
            "trust_score": 0.85,
            "modality": "text",
            "asset_id": "asset_pedagogy",
        },
        {
            "id": "doc_017",
            "title": "Historical Data Archive 1998",
            "text": "# Historical Archive\nA 1998 index process used manual card catalogs and weekly reconciliation. Digitization later replaced physical lookup constraints.",
            "timestamp": "2023-06-12",
            "trust_score": 0.82,
            "modality": "text",
            "asset_id": "asset_archive",
        },
        {
            "id": "doc_018",
            "title": "AI Ethics Topic Note",
            "text": "AI ethics includes fairness audits, model transparency, and accountability controls. Governance frameworks align deployment with policy obligations.",
            "timestamp": "2025-09-12",
            "trust_score": 0.90,
            "modality": "text",
            "asset_id": "asset_ai_ethics",
        },
        {
            "id": "doc_019",
            "title": "Medieval Trade Grade 10",
            "text": "# Medieval Trade\n## Grade 10\nTrade routes connected towns through fairs and merchant guilds. Students focus on basic exchange flows and goods movement.",
            "timestamp": "2025-01-20",
            "trust_score": 0.88,
            "modality": "text",
            "asset_id": "asset_trade",
            "grade_level": "grade_10",
        },
        {
            "id": "doc_020",
            "title": "Medieval Trade Grade 12",
            "text": "# Medieval Trade\n## Grade 12\nAdvanced treatment examines credit instruments, maritime insurance, and institutional effects on early capitalism.",
            "timestamp": "2025-01-22",
            "trust_score": 0.89,
            "modality": "text",
            "asset_id": "asset_trade",
            "grade_level": "grade_12",
        },
    ]


def default_query_records() -> List[Dict[str, Any]]:
    return [
        {
            "query": "Compare the treatment of the Industrial Revolution in Grade 10 vs Grade 12.",
            "relevant_ids": ["doc_001", "doc_002"],
        },
        {
            "query": "What caused the French Revolution?",
            "relevant_ids": ["doc_003"],
        },
        {
            "query": "Explain why the Civil Rights Movement mattered.",
            "relevant_ids": ["doc_005"],
        },
        {
            "query": "What are the latest renewable energy policy updates?",
            "relevant_ids": ["doc_006", "doc_013"],
        },
        {
            "query": "Find a trustworthy source about the recent cybersecurity incident.",
            "relevant_ids": ["doc_008"],
        },
        {
            "query": "What evidence exists across image and audio about steam engines?",
            "relevant_ids": ["doc_009", "doc_010"],
        },
        {
            "query": "Which scene discusses factory floor conditions?",
            "relevant_ids": ["doc_011"],
        },
        {
            "query": "Give steps to archive historical documents.",
            "relevant_ids": ["doc_012"],
        },
        {
            "query": "Why did supply chain delays happen?",
            "relevant_ids": ["doc_015"],
        },
        {
            "query": "Find Grade 12 material on medieval trade.",
            "relevant_ids": ["doc_020"],
        },
    ]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_corpus(corpus_path: Optional[str | Path] = None) -> List[Dict[str, Any]]:
    if corpus_path is None:
        return default_corpus_records()
    path = Path(corpus_path)
    if not path.exists():
        return default_corpus_records()
    return load_jsonl(path)


def load_queries(query_path: Optional[str | Path] = None) -> List[Dict[str, Any]]:
    if query_path is None:
        return default_query_records()
    path = Path(query_path)
    if not path.exists():
        return default_query_records()
    return load_jsonl(path)


class IndexingMethodRunner:
    def __init__(self, method_id: str) -> None:
        if method_id not in METHOD_SPEC_MAP:
            raise ValueError(f"Unknown method_id: {method_id}")
        self.spec = METHOD_SPEC_MAP[method_id]
        self.method_id = method_id
        self.documents: List[Dict[str, Any]] = []
        self.chunks: List[Chunk] = []
        self.chunk_lookup: Dict[str, Chunk] = {}

        self.embedder: Optional[DenseEmbedder] = None
        self.embedder_v2: Optional[DenseEmbedder] = None
        self.embeddings: Optional[np.ndarray] = None
        self.embeddings_v2: Optional[np.ndarray] = None

        self.bm25: Any = None
        self.inverted: Dict[str, set[int]] = defaultdict(set)
        self.term_freqs: List[Counter[str]] = []

        self.cluster_model: Optional[KMeans] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_centers: Optional[np.ndarray] = None

        self.hash_planes: Optional[np.ndarray] = None
        self.hash_codes: Optional[np.ndarray] = None

        self.summary_embedder: Optional[DenseEmbedder] = None
        self.summary_embeddings: Optional[np.ndarray] = None
        self.summaries: List[str] = []

        self.entity_index: Dict[str, set[int]] = defaultdict(set)
        self.entity_graph: Any = None
        self.semantic_graph: Dict[int, set[int]] = defaultdict(set)
        self.causal_graph: Dict[int, set[int]] = defaultdict(set)

        self.parent_map: Dict[str, str] = {}
        self.parent_chunk_lookup: Dict[str, Chunk] = {}

        self.trust_scores: Optional[np.ndarray] = None
        self.age_days: Optional[np.ndarray] = None
        self.variance_scores: Optional[np.ndarray] = None
        self.confidence_scores: Optional[np.ndarray] = None
        self.outlier_mask: Optional[np.ndarray] = None
        self.drift_weights: Optional[np.ndarray] = None

        self.feedback_bias: Dict[str, float] = defaultdict(float)
        self.runtime_config: Dict[str, Any] = {}

    def build(self, documents: Sequence[Dict[str, Any]]) -> None:
        self.documents = [dict(doc) for doc in documents]
        self.chunks = self._build_chunks(self.documents)
        self.chunk_lookup = {chunk.chunk_id: chunk for chunk in self.chunks}
        texts = [chunk.text for chunk in self.chunks]

        self.trust_scores = np.array([
            safe_float(chunk.metadata.get("trust_score", 0.8), 0.8) for chunk in self.chunks
        ], dtype=np.float32)

        today = date.today()
        age_days: List[float] = []
        for chunk in self.chunks:
            raw_ts = chunk.metadata.get("timestamp")
            ts_str: Optional[str]
            if isinstance(raw_ts, date):
                ts_str = raw_ts.isoformat()
            elif raw_ts is None:
                ts_str = None
            else:
                ts_str = str(raw_ts)

            parsed_ts = parse_date(ts_str)
            if parsed_ts is None:
                age_days.append(3650.0)
            else:
                age_days.append(float((today - parsed_ts).days))

        self.age_days = np.array(age_days, dtype=np.float32)

        variance_scores = np.array([self._estimate_variance(chunk.text) for chunk in self.chunks], dtype=np.float32)
        self.variance_scores = variance_scores
        self.confidence_scores = np.exp(-2.0 * variance_scores).astype(np.float32)

        self.embedder = DenseEmbedder()
        self.embeddings = self.embedder.fit(texts)

        if self.method_id == "11.3_continual_re_embedding_index":
            self.embedder_v2 = DenseEmbedder(ngram_range=(1, 3))
            self.embeddings_v2 = self.embedder_v2.fit(texts)

        self._build_sparse_structures(texts)
        self._build_method_specific_structures(texts)

    def _build_chunks(self, documents: Sequence[Dict[str, Any]]) -> List[Chunk]:
        method = self.method_id
        chunks: List[Chunk] = []

        if method == "11.1_self_tuning_index":
            lengths = [len(tokenize(doc.get("text", ""))) for doc in documents]
            median_len = float(np.median(lengths)) if lengths else 120.0
            if median_len < 120:
                self.runtime_config["chunk_size"] = 80
            elif median_len < 220:
                self.runtime_config["chunk_size"] = 120
            else:
                self.runtime_config["chunk_size"] = 200

        for doc in documents:
            doc_id = str(doc.get("id", f"doc_{len(chunks)}"))
            text = str(doc.get("text", "")).strip()
            if not text:
                continue
            base_meta = {
                "title": doc.get("title", ""),
                "timestamp": doc.get("timestamp"),
                "trust_score": safe_float(doc.get("trust_score", 0.8), 0.8),
                "modality": doc.get("modality", "text"),
                "asset_id": doc.get("asset_id"),
                "grade_level": doc.get("grade_level"),
            }

            if method == "1.2_sliding_window_overlap":
                pieces = chunk_sliding(text, chunk_size=120, overlap=40)
                for i, piece in enumerate(pieces):
                    chunks.append(Chunk(f"{doc_id}::c{i}", doc_id, piece, dict(base_meta, chunk_strategy="sliding")))

            elif method == "1.3_adaptive_semantic_chunking":
                pieces = chunk_semantic_adaptive(text)
                for i, piece in enumerate(pieces):
                    chunks.append(Chunk(f"{doc_id}::c{i}", doc_id, piece, dict(base_meta, chunk_strategy="adaptive")))

            elif method in {"1.4_query_aware_dynamic_chunking", "1.6_sentence_level_indexing"}:
                pieces = chunk_sentences(text)
                for i, piece in enumerate(pieces):
                    chunks.append(Chunk(f"{doc_id}::c{i}", doc_id, piece, dict(base_meta, sentence_index=i, chunk_strategy="sentence")))

            elif method == "1.5_small_to_big_hierarchical_chunking":
                small = chunk_fixed(text, chunk_size=60)
                parent_size = 3
                for parent_i in range(0, len(small), parent_size):
                    parent_id = f"{doc_id}::p{parent_i // parent_size}"
                    parent_text = " ".join(small[parent_i : parent_i + parent_size])
                    parent_chunk = Chunk(parent_id, doc_id, parent_text, dict(base_meta, level="parent"))
                    self.parent_chunk_lookup[parent_id] = parent_chunk
                    for child_offset, child_text in enumerate(small[parent_i : parent_i + parent_size]):
                        child_i = parent_i + child_offset
                        child_id = f"{doc_id}::c{child_i}"
                        self.parent_map[child_id] = parent_id
                        chunks.append(Chunk(child_id, doc_id, child_text, dict(base_meta, level="child", parent_id=parent_id)))

            elif method in {"4.1_tree_structured_index", "4.2_summary_augmented_index", "4.3_structural_aware_index"}:
                blocks = chunk_structural(text)
                for i, (heading, depth, body) in enumerate(blocks):
                    meta = dict(base_meta, heading=heading, depth=depth, chunk_strategy="structural")
                    chunks.append(Chunk(f"{doc_id}::c{i}", doc_id, body, meta))

            elif method == "10.3_scene_aware_media_index":
                if str(doc.get("modality", "text")).lower() == "video":
                    scene_blocks = extract_scene_chunks(text)
                    for i, (scene_name, scene_text) in enumerate(scene_blocks):
                        meta = dict(base_meta, scene=scene_name, chunk_strategy="scene")
                        chunks.append(Chunk(f"{doc_id}::scene{i}", doc_id, scene_text, meta))
                else:
                    chunks.append(Chunk(f"{doc_id}::c0", doc_id, text, dict(base_meta, chunk_strategy="full")))

            else:
                chunk_size = int(self.runtime_config.get("chunk_size", 120))
                pieces = chunk_fixed(text, chunk_size=chunk_size)
                for i, piece in enumerate(pieces):
                    if method in {"10.1_joint_embedding_index", "10.2_cross_modal_linked_index"}:
                        modality = str(base_meta.get("modality", "text"))
                        piece = f"[{modality}] {piece}"
                    chunks.append(Chunk(f"{doc_id}::c{i}", doc_id, piece, dict(base_meta, chunk_strategy="fixed")))

        return chunks

    def _build_sparse_structures(self, texts: Sequence[str]) -> None:
        tokenized = [tokenize(t) for t in texts]
        self.term_freqs = [Counter(tokens) for tokens in tokenized]
        for idx, tokens in enumerate(tokenized):
            for token in set(tokens):
                self.inverted[token].add(idx)

        if BM25Okapi is not None:
            self.bm25 = BM25Okapi(tokenized)

    def _build_method_specific_structures(self, texts: Sequence[str]) -> None:
        if self.method_id in {
            "2.2_approximate_nearest_neighbor_index",
            "2.4_hybrid_dense_partition_index",
            "6.1_topic_partition_index",
        }:
            self._fit_clusters()

        if self.method_id == "6.2_semantic_hashing_index":
            self._fit_semantic_hashing()

        if self.method_id == "4.2_summary_augmented_index":
            self.summaries = [self._summarize_text(t) for t in texts]
            self.summary_embedder = DenseEmbedder()
            self.summary_embeddings = self.summary_embedder.fit(self.summaries)

        if self.method_id in {
            "5.1_knowledge_graph_index",
            "5.2_entity_centric_index",
            "5.4_multi_hop_retrieval_graph",
        }:
            self._build_entity_index_graph()

        if self.method_id in {"5.3_semantic_similarity_graph", "5.4_multi_hop_retrieval_graph"}:
            self._build_semantic_graph(k=4)

        if self.method_id == "8.2_outlier_aware_vector_index":
            self._build_outlier_mask()

        if self.method_id == "8.3_drift_aware_index":
            self._build_drift_weights()

        if self.method_id == "12.1_causal_graph_index":
            self._build_causal_graph()

    def _fit_clusters(self, k: int = 6) -> None:
        if self.embeddings is None or len(self.embeddings) == 0:
            return
        n_samples = len(self.embeddings)
        n_clusters = max(2, min(k, n_samples))
        model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        self.cluster_labels = model.fit_predict(self.embeddings)
        self.cluster_centers = model.cluster_centers_.astype(np.float32)
        self.cluster_model = model

    def _fit_semantic_hashing(self, bits: int = 32) -> None:
        if self.embeddings is None:
            return
        rng = np.random.default_rng(42)
        self.hash_planes = rng.normal(0.0, 1.0, size=(self.embeddings.shape[1], bits)).astype(np.float32)
        projections = np.dot(self.embeddings, self.hash_planes)
        self.hash_codes = (projections > 0).astype(np.int8)

    def _summarize_text(self, text: str) -> str:
        sents = split_sentences(text)
        if not sents:
            return text[:160]
        if len(sents) == 1:
            return sents[0]
        return f"{sents[0]} {sents[-1]}"

    def _build_entity_index_graph(self) -> None:
        for idx, chunk in enumerate(self.chunks):
            for entity in extract_entities(chunk.text):
                key = entity.lower()
                self.entity_index[key].add(idx)

        if self.method_id == "5.1_knowledge_graph_index" and nx is not None:
            graph = nx.Graph()
            for idx, chunk in enumerate(self.chunks):
                cnode = f"chunk::{idx}"
                graph.add_node(cnode, node_type="chunk")
                ents = extract_entities(chunk.text)
                for ent in ents:
                    enode = f"entity::{ent.lower()}"
                    graph.add_node(enode, node_type="entity")
                    graph.add_edge(cnode, enode)
            self.entity_graph = graph

    def _build_semantic_graph(self, k: int = 4) -> None:
        if self.embeddings is None:
            return
        sims = np.dot(self.embeddings, self.embeddings.T)
        np.fill_diagonal(sims, -np.inf)
        for i in range(len(self.chunks)):
            neighbors = np.argsort(-sims[i])[:k]
            for n in neighbors:
                self.semantic_graph[i].add(int(n))
                self.semantic_graph[int(n)].add(i)

    def _build_outlier_mask(self) -> None:
        if self.embeddings is None or len(self.embeddings) < 8:
            self.outlier_mask = np.ones(len(self.chunks), dtype=bool)
            return
        model = IsolationForest(contamination=0.08, random_state=42)
        labels = model.fit_predict(self.embeddings)
        self.outlier_mask = labels > 0

    def _build_drift_weights(self) -> None:
        if self.embeddings is None:
            return
        center = np.mean(self.embeddings, axis=0)
        dists = np.linalg.norm(self.embeddings - center, axis=1)
        if np.max(dists) <= 1e-6:
            self.drift_weights = np.ones_like(dists)
            return
        norm = dists / (np.max(dists) + 1e-6)
        self.drift_weights = (1.0 - 0.5 * norm).astype(np.float32)

    def _build_causal_graph(self) -> None:
        markers = ["because", "led to", "resulted in", "therefore", "caused"]
        marker_to_chunks: Dict[str, List[int]] = defaultdict(list)
        for idx, chunk in enumerate(self.chunks):
            text = chunk.text.lower()
            for marker in markers:
                if marker in text:
                    marker_to_chunks[marker].append(idx)
        for chunk_ids in marker_to_chunks.values():
            for i in chunk_ids:
                for j in chunk_ids:
                    if i != j:
                        self.causal_graph[i].add(j)

    def _estimate_variance(self, text: str) -> float:
        toks = tokenize(text)
        if not toks:
            return 1.0
        uniq = len(set(toks))
        diversity = uniq / len(toks)
        length_penalty = 1.0 / math.sqrt(len(toks))
        return float(0.6 * (1.0 - diversity) + 0.4 * length_penalty)

    def _encode_query(self, query: str) -> np.ndarray:
        if self.embedder is None:
            raise RuntimeError("Embedder not initialized. Call build() first.")
        return self.embedder.encode([query])[0]

    def _dense_scores(self, query: str, use_v2: bool = False) -> np.ndarray:
        if self.embeddings is None:
            raise RuntimeError("Embeddings not initialized. Call build() first.")
        if use_v2:
            if self.embedder_v2 is None or self.embeddings_v2 is None:
                return np.dot(self.embeddings, self._encode_query(query))
            q = self.embedder_v2.encode([query])[0]
            return np.dot(self.embeddings_v2, q)
        q = self._encode_query(query)
        return np.dot(self.embeddings, q)

    def _bm25_scores(self, query: str) -> np.ndarray:
        if not self.chunks:
            return np.array([], dtype=np.float32)
        if self.bm25 is not None:
            scores = np.array(self.bm25.get_scores(tokenize(query)), dtype=np.float32)
            return scores

        # Fallback: manual tf-idf lexical scoring.
        query_tokens = tokenize(query)
        n_docs = len(self.chunks)
        scores = np.zeros(n_docs, dtype=np.float32)
        for token in query_tokens:
            postings = self.inverted.get(token)
            if not postings:
                continue
            df = len(postings)
            idf = math.log((n_docs + 1.0) / (df + 1.0)) + 1.0
            for idx in postings:
                scores[idx] += idf * self.term_freqs[idx].get(token, 0)
        return scores

    def _inverted_scores(self, query: str) -> np.ndarray:
        query_tokens = tokenize(query)
        n_docs = len(self.chunks)
        scores = np.zeros(n_docs, dtype=np.float32)
        for token in query_tokens:
            postings = self.inverted.get(token)
            if not postings:
                continue
            df = len(postings)
            idf = math.log((n_docs + 1.0) / (df + 1.0)) + 1.0
            for idx in postings:
                tf = self.term_freqs[idx].get(token, 0)
                scores[idx] += tf * idf
        return scores

    def _cluster_candidate_mask(self, query: str, top_clusters: int = 2) -> np.ndarray:
        if self.cluster_centers is None or self.cluster_labels is None:
            return np.ones(len(self.chunks), dtype=bool)
        q = self._encode_query(query)
        centroid_scores = np.dot(self.cluster_centers, q)
        keep_clusters = set(np.argsort(-centroid_scores)[:top_clusters].tolist())
        return np.array([int(label) in keep_clusters for label in self.cluster_labels], dtype=bool)

    def _semantic_hash_candidates(self, query: str, threshold: int = 8) -> np.ndarray:
        if self.hash_planes is None or self.hash_codes is None or self.embeddings is None:
            return np.ones(len(self.chunks), dtype=bool)
        q = self._encode_query(query)
        q_code = (np.dot(q, self.hash_planes) > 0).astype(np.int8)
        distances = np.sum(np.abs(self.hash_codes - q_code), axis=1)
        if np.min(distances) > threshold:
            keep = np.argsort(distances)[: min(20, len(distances))]
            mask = np.zeros(len(distances), dtype=bool)
            mask[keep] = True
            return mask
        return distances <= threshold

    def _score_query_aware(self, query: str) -> np.ndarray:
        base = self._dense_scores(query)
        q_tokens = set(tokenize(query))
        scores = base.copy()
        for idx, chunk in enumerate(self.chunks):
            c_tokens = set(tokenize(chunk.text))
            overlap = len(q_tokens & c_tokens)
            scores[idx] += 0.2 * overlap

            # Add local sentence-window context from neighboring sentence chunks in same doc.
            sentence_idx = chunk.metadata.get("sentence_index")
            if sentence_idx is not None:
                prev_id = f"{chunk.doc_id}::c{int(sentence_idx) - 1}"
                next_id = f"{chunk.doc_id}::c{int(sentence_idx) + 1}"
                for neighbor_id in (prev_id, next_id):
                    n_chunk = self.chunk_lookup.get(neighbor_id)
                    if n_chunk:
                        n_tokens = set(tokenize(n_chunk.text))
                        scores[idx] += 0.05 * len(q_tokens & n_tokens)
        return scores

    def _apply_graph_expansion(self, scores: np.ndarray, hops: int = 1, decay: float = 0.2) -> np.ndarray:
        if not self.semantic_graph:
            return scores
        expanded = scores.copy()
        seeds = np.argsort(-scores)[: min(6, len(scores))]
        frontier = {(int(seed), 0) for seed in seeds}
        visited = set(int(seed) for seed in seeds)

        while frontier:
            node, depth = frontier.pop()
            if depth >= hops:
                continue
            for nbr in self.semantic_graph.get(node, set()):
                expanded[nbr] += max(0.0, scores[node]) * (decay ** (depth + 1))
                if nbr not in visited:
                    visited.add(nbr)
                    frontier.add((nbr, depth + 1))
        return expanded

    def _apply_entity_boost(self, query: str, base_scores: np.ndarray, weight: float = 0.35) -> np.ndarray:
        entities = [e.lower() for e in extract_entities(query)]
        if not entities:
            return base_scores
        scores = base_scores.copy()
        for ent in entities:
            for idx in self.entity_index.get(ent, set()):
                scores[idx] += weight
        return scores

    def _scores_for_method(self, query: str) -> np.ndarray:
        m = self.method_id

        if m == "3.1_inverted_index":
            return self._inverted_scores(query)

        if m == "3.2_sparse_dense_fusion_index":
            dense = normalize_scores(self._dense_scores(query))
            sparse = normalize_scores(self._bm25_scores(query))
            return 0.55 * dense + 0.45 * sparse

        if m == "3.3_weighted_term_index":
            return self._bm25_scores(query)

        scores = self._dense_scores(query)

        if m == "1.4_query_aware_dynamic_chunking":
            scores = self._score_query_aware(query)

        elif m == "2.2_approximate_nearest_neighbor_index":
            mask = self._cluster_candidate_mask(query, top_clusters=2)
            scores = np.where(mask, scores, -np.inf)

        elif m == "2.3_quantized_vector_index":
            if self.embeddings is not None:
                q = self._encode_query(query).astype(np.float16)
                approx = self.embeddings.astype(np.float16)
                scores = np.dot(approx, q).astype(np.float32)

        elif m == "2.4_hybrid_dense_partition_index":
            mask = self._cluster_candidate_mask(query, top_clusters=3)
            masked_scores = np.where(mask, scores, -np.inf)
            cluster_prior = np.zeros_like(scores)
            if self.cluster_labels is not None and self.cluster_centers is not None:
                q = self._encode_query(query)
                centroid_scores = np.dot(self.cluster_centers, q)
                norm_centroids = normalize_scores(centroid_scores)
                for i, label in enumerate(self.cluster_labels):
                    cluster_prior[i] = norm_centroids[int(label)]
            scores = 0.7 * normalize_scores(masked_scores) + 0.3 * cluster_prior

        elif m == "4.1_tree_structured_index":
            depth_boost = np.array([
                1.0 / (1.0 + safe_float(chunk.metadata.get("depth", 0), 0.0))
                for chunk in self.chunks
            ], dtype=np.float32)
            scores = scores * (0.7 + 0.3 * depth_boost)

        elif m == "4.2_summary_augmented_index":
            if self.summary_embedder is not None and self.summary_embeddings is not None:
                q = self.summary_embedder.encode([query])[0]
                summary_scores = np.dot(self.summary_embeddings, q)
                scores = 0.4 * normalize_scores(scores) + 0.6 * normalize_scores(summary_scores)

        elif m == "4.3_structural_aware_index":
            q_tokens = set(tokenize(query))
            structural_bonus = []
            for chunk in self.chunks:
                heading = str(chunk.metadata.get("heading", "")).lower()
                bonus = 0.0
                for token in q_tokens:
                    if token in heading:
                        bonus += 0.15
                structural_bonus.append(bonus)
            scores = scores + np.array(structural_bonus, dtype=np.float32)

        elif m == "5.1_knowledge_graph_index":
            scores = self._apply_entity_boost(query, scores, weight=0.5)

        elif m == "5.2_entity_centric_index":
            entity_scores = self._apply_entity_boost(query, np.zeros_like(scores), weight=1.0)
            if float(np.max(entity_scores)) > 0:
                scores = 0.25 * normalize_scores(scores) + 0.75 * normalize_scores(entity_scores)

        elif m == "5.3_semantic_similarity_graph":
            scores = self._apply_graph_expansion(scores, hops=1, decay=0.25)

        elif m == "5.4_multi_hop_retrieval_graph":
            scores = self._apply_entity_boost(query, scores, weight=0.25)
            scores = self._apply_graph_expansion(scores, hops=2, decay=0.25)

        elif m == "6.1_topic_partition_index":
            mask = self._cluster_candidate_mask(query, top_clusters=1)
            scores = np.where(mask, scores, -np.inf)

        elif m == "6.2_semantic_hashing_index":
            mask = self._semantic_hash_candidates(query)
            scores = np.where(mask, scores, -np.inf)

        elif m == "7.1_time_decayed_index":
            if self.age_days is not None:
                decay = np.exp(-self.age_days / 365.0)
                scores = scores * decay

        elif m == "7.2_sliding_temporal_window_index":
            if self.age_days is not None:
                recent = self.age_days <= 180
                weights = np.where(recent, 1.1, 0.8)
                scores = scores * weights

        elif m == "8.1_trust_weighted_index":
            if self.trust_scores is not None:
                scores = scores * self.trust_scores

        elif m == "8.2_outlier_aware_vector_index":
            if self.outlier_mask is not None:
                scores = np.where(self.outlier_mask, scores, -np.inf)

        elif m == "8.3_drift_aware_index":
            if self.drift_weights is not None:
                scores = scores * self.drift_weights

        elif m == "9.1_distributional_embedding_index":
            if self.variance_scores is not None:
                scores = scores / (1.0 + self.variance_scores)

        elif m == "9.2_confidence_propagating_index":
            if self.confidence_scores is not None:
                scores = scores * self.confidence_scores

        elif m == "10.1_joint_embedding_index":
            # Joint space is handled by modality-prefixed chunk text in _build_chunks.
            pass

        elif m == "10.2_cross_modal_linked_index":
            # Base dense retrieval; cross-modal link expansion happens in result assembly.
            pass

        elif m == "10.3_scene_aware_media_index":
            q_tokens = set(tokenize(query))
            scene_bonus = []
            for chunk in self.chunks:
                scene = str(chunk.metadata.get("scene", "")).lower()
                bonus = 0.2 if any(tok in scene for tok in q_tokens) else 0.0
                scene_bonus.append(bonus)
            scores = scores + np.array(scene_bonus, dtype=np.float32)

        elif m == "11.2_feedback_driven_index":
            bias = np.array([self.feedback_bias.get(chunk.chunk_id, 0.0) for chunk in self.chunks], dtype=np.float32)
            scores = scores + bias

        elif m == "11.3_continual_re_embedding_index":
            v1 = normalize_scores(scores)
            v2 = normalize_scores(self._dense_scores(query, use_v2=True))
            scores = 0.35 * v1 + 0.65 * v2

        elif m == "12.1_causal_graph_index":
            query_lower = query.lower()
            markers = ["because", "cause", "caused", "led to", "why"]
            if any(mk in query_lower for mk in markers):
                boosted = scores.copy()
                seeds = np.argsort(-scores)[: min(5, len(scores))]
                for seed in seeds:
                    for nbr in self.causal_graph.get(int(seed), set()):
                        boosted[nbr] += 0.2
                scores = boosted

        elif m == "12.2_explanation_aware_index":
            q = query.lower()
            is_expl = any(token in q for token in ["why", "explain", "reason", "because"])
            if is_expl:
                bonus = np.array([
                    0.15
                    if any(marker in chunk.text.lower() for marker in ["because", "therefore", "this means", "as a result"])
                    else 0.0
                    for chunk in self.chunks
                ], dtype=np.float32)
                scores = scores + bonus

        elif m == "12.3_plan_oriented_index":
            q = query.lower()
            wants_steps = any(token in q for token in ["how", "steps", "procedure", "plan"])
            if wants_steps:
                bonus = np.array([
                    0.18
                    if any(marker in chunk.text.lower() for marker in ["step 1", "step 2", "first", "then", "procedure"])
                    else 0.0
                    for chunk in self.chunks
                ], dtype=np.float32)
                scores = scores + bonus

        return scores

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.chunks:
            return []

        scores = self._scores_for_method(query)
        valid = np.isfinite(scores)
        if not valid.any():
            return []

        order = np.argsort(-scores)
        results: List[Dict[str, Any]] = []

        for idx in order:
            idx = int(idx)
            if not np.isfinite(scores[idx]):
                continue
            chunk = self.chunks[idx]
            item = {
                "rank": len(results) + 1,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "score": float(scores[idx]),
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            results.append(item)
            if len(results) >= max(top_k * 2, 10):
                break

        if self.method_id == "1.5_small_to_big_hierarchical_chunking":
            merged: Dict[str, Dict[str, Any]] = {}
            for row in results:
                parent_id = self.parent_map.get(row["chunk_id"])
                if not parent_id:
                    continue
                parent_chunk = self.parent_chunk_lookup.get(parent_id)
                if not parent_chunk:
                    continue
                prev = merged.get(parent_id)
                if prev is None or row["score"] > prev["score"]:
                    merged[parent_id] = {
                        "rank": len(merged) + 1,
                        "chunk_id": parent_chunk.chunk_id,
                        "doc_id": parent_chunk.doc_id,
                        "score": row["score"],
                        "text": parent_chunk.text,
                        "metadata": parent_chunk.metadata,
                    }
            results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)

        if self.method_id == "10.2_cross_modal_linked_index":
            by_asset: Dict[str, Dict[str, Any]] = {}
            for row in results:
                asset_id = str(row["metadata"].get("asset_id") or "")
                if not asset_id:
                    continue
                best = by_asset.get(asset_id)
                if best is None or row["score"] > best["score"]:
                    by_asset[asset_id] = row

            expanded = list(results)
            for asset_id, best_row in by_asset.items():
                for chunk in self.chunks:
                    if str(chunk.metadata.get("asset_id") or "") != asset_id:
                        continue
                    if any(existing["chunk_id"] == chunk.chunk_id for existing in expanded):
                        continue
                    expanded.append(
                        {
                            "rank": len(expanded) + 1,
                            "chunk_id": chunk.chunk_id,
                            "doc_id": chunk.doc_id,
                            "score": float(best_row["score"] * 0.92),
                            "text": chunk.text,
                            "metadata": chunk.metadata,
                        }
                    )
            results = sorted(expanded, key=lambda x: x["score"], reverse=True)

        # Deduplicate by doc for final top-k list.
        final_rows: List[Dict[str, Any]] = []
        seen_docs = set()
        for row in results:
            doc_id = row["doc_id"]
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            row["rank"] = len(final_rows) + 1
            final_rows.append(row)
            if len(final_rows) >= top_k:
                break
        return final_rows

    def evaluate(self, queries: Sequence[Dict[str, Any]], top_k: int = 5) -> Dict[str, Any]:
        per_query: List[Dict[str, Any]] = []

        p_scores = []
        r_scores = []
        mrr_scores = []
        ndcg_scores = []

        for item in queries:
            query = str(item.get("query", ""))
            relevant = set(str(x) for x in item.get("relevant_ids", []))
            results = self.search(query, top_k=top_k)
            ranked_ids = [row["doc_id"] for row in results]

            hits = [doc_id for doc_id in ranked_ids if doc_id in relevant]
            precision = len(hits) / max(1, top_k)
            recall = len(hits) / max(1, len(relevant))

            rr = 0.0
            for idx, doc_id in enumerate(ranked_ids, start=1):
                if doc_id in relevant:
                    rr = 1.0 / idx
                    break

            dcg = 0.0
            for idx, doc_id in enumerate(ranked_ids, start=1):
                rel = 1.0 if doc_id in relevant else 0.0
                dcg += rel / math.log2(idx + 1)
            ideal_hits = min(len(relevant), top_k)
            idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
            ndcg = dcg / idcg if idcg > 0 else 0.0

            p_scores.append(precision)
            r_scores.append(recall)
            mrr_scores.append(rr)
            ndcg_scores.append(ndcg)

            per_query.append(
                {
                    "query": query,
                    "relevant_ids": sorted(relevant),
                    "retrieved_ids": ranked_ids,
                    "precision_at_k": precision,
                    "recall_at_k": recall,
                    "mrr": rr,
                    "ndcg_at_k": ndcg,
                }
            )

            if self.method_id == "11.2_feedback_driven_index":
                for row in results:
                    if row["doc_id"] in relevant:
                        self.feedback_bias[row["chunk_id"]] += 0.05

        metrics = {
            "precision_at_k": float(np.mean(p_scores) if p_scores else 0.0),
            "recall_at_k": float(np.mean(r_scores) if r_scores else 0.0),
            "mrr": float(np.mean(mrr_scores) if mrr_scores else 0.0),
            "ndcg_at_k": float(np.mean(ndcg_scores) if ndcg_scores else 0.0),
        }

        return {
            "method_id": self.spec.method_id,
            "method_name": self.spec.name,
            "category": self.spec.category,
            "description": self.spec.description,
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "metrics": metrics,
            "runtime_config": self.runtime_config,
            "per_query": per_query,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


def run_method(
    method_id: str,
    corpus_path: Optional[str | Path] = None,
    queries_path: Optional[str | Path] = None,
    output_dir: str | Path = "results",
    top_k: int = 5,
) -> Dict[str, Any]:
    runner = IndexingMethodRunner(method_id)
    corpus = load_corpus(corpus_path)
    queries = load_queries(queries_path)
    runner.build(corpus)
    result = runner.evaluate(queries, top_k=top_k)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{method_slug(method_id)}.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    result["result_path"] = str(out_path)
    return result


def run_all_methods(
    corpus_path: Optional[str | Path] = None,
    queries_path: Optional[str | Path] = None,
    output_dir: str | Path = "results",
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    outputs = []
    for spec in METHOD_SPECS:
        outputs.append(
            run_method(
                method_id=spec.method_id,
                corpus_path=corpus_path,
                queries_path=queries_path,
                output_dir=output_dir,
                top_k=top_k,
            )
        )
    return outputs


def _main() -> None:
    parser = argparse.ArgumentParser(description="Run indexing method experiments.")
    parser.add_argument("--method", dest="method_id", default="1.1_fixed_length_chunking")
    parser.add_argument("--corpus", dest="corpus_path", default=None)
    parser.add_argument("--queries", dest="queries_path", default=None)
    parser.add_argument("--output-dir", dest="output_dir", default="results")
    parser.add_argument("--top-k", dest="top_k", type=int, default=5)
    parser.add_argument("--all", dest="run_all", action="store_true")
    args = parser.parse_args()

    if args.run_all:
        rows = run_all_methods(
            corpus_path=args.corpus_path,
            queries_path=args.queries_path,
            output_dir=args.output_dir,
            top_k=args.top_k,
        )
        print(json.dumps({"methods": len(rows), "output_dir": args.output_dir}, indent=2))
        return

    result = run_method(
        method_id=args.method_id,
        corpus_path=args.corpus_path,
        queries_path=args.queries_path,
        output_dir=args.output_dir,
        top_k=args.top_k,
    )
    print(json.dumps({"method_id": result["method_id"], "metrics": result["metrics"], "result_path": result["result_path"]}, indent=2))


if __name__ == "__main__":
    _main()
