from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import IsolationForest

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    SemanticSplitterNodeParser,
    SentenceSplitter,
    SentenceWindowNodeParser,
    TokenTextSplitter,
    get_leaf_nodes,
)
from llama_index.core.schema import BaseNode, NodeRelationship, TextNode

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
except Exception:
    HuggingFaceEmbedding = None

try:
    from llama_index.retrievers.bm25 import BM25Retriever
except Exception:
    BM25Retriever = None

try:
    import networkx as nx
except Exception:
    nx = None


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
class ScoredNode:
    node_id: str
    doc_id: str
    score: float
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


def node_text(node: BaseNode) -> str:
    try:
        return node.get_content(metadata_mode="none")
    except Exception:
        try:
            return str(node.text)
        except Exception:
            return ""


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


def _configure_embed_models() -> Tuple[Any, Any, str]:
    preferred = os.getenv("LLAMAINDEX_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    force_mock = os.getenv("LLAMAINDEX_FORCE_MOCK", "0") == "1"
    embed_model = None
    v2_model = None
    source = "mock"

    if not force_mock and HuggingFaceEmbedding is not None:
        try:
            embed_model = HuggingFaceEmbedding(model_name=preferred)
            source = preferred
        except Exception:
            embed_model = None

        if embed_model is not None:
            try:
                v2_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
            except Exception:
                v2_model = None

    if embed_model is None:
        embed_model = MockEmbedding(embed_dim=384)
        source = "mock"

    if v2_model is None:
        v2_model = embed_model

    Settings.embed_model = embed_model
    return embed_model, v2_model, source


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


class LlamaIndexMethodRunner:
    _CAUSAL_QUERY_MARKERS = ("because", "cause", "caused", "led to", "why")
    _EXPLANATION_QUERY_MARKERS = ("why", "explain", "reason", "because")
    _EXPLANATION_TEXT_MARKERS = ("because", "therefore", "this means", "as a result")
    _PLAN_QUERY_MARKERS = ("how", "steps", "procedure", "plan")
    _PLAN_TEXT_MARKERS = ("step 1", "step 2", "first", "then", "procedure")

    def __init__(self, method_id: str) -> None:
        if method_id not in METHOD_SPEC_MAP:
            raise ValueError(f"Unknown method_id: {method_id}")

        self.spec = METHOD_SPEC_MAP[method_id]
        self.method_id = method_id
        self.documents: List[Dict[str, Any]] = []

        self.storage_context: Optional[StorageContext] = None
        self.all_nodes: List[BaseNode] = []
        self.index_nodes: List[BaseNode] = []
        self.node_lookup: Dict[str, BaseNode] = {}
        self.node_id_to_idx: Dict[str, int] = {}

        self.vector_index: Optional[VectorStoreIndex] = None
        self.dense_retriever: Any = None
        self.bm25_retriever: Any = None

        self.embed_model: Any = None
        self.embed_model_v2: Any = None
        self.embed_source: str = ""

        self.node_embeddings: Optional[np.ndarray] = None
        self.node_embeddings_v2: Optional[np.ndarray] = None

        self.inverted: Dict[str, set[int]] = defaultdict(set)
        self.term_freqs: List[Counter[str]] = []

        self.cluster_labels: Optional[np.ndarray] = None
        self.cluster_centers: Optional[np.ndarray] = None
        self.hash_planes: Optional[np.ndarray] = None
        self.hash_codes: Optional[np.ndarray] = None

        self.summaries: List[str] = []
        self.summary_embeddings: Optional[np.ndarray] = None

        self.entity_index: Dict[str, set[int]] = defaultdict(set)
        self.entity_graph: Any = None
        self.semantic_graph: Dict[int, set[int]] = defaultdict(set)
        self.causal_graph: Dict[int, set[int]] = defaultdict(set)

        self.parent_map: Dict[str, str] = {}

        self.trust_scores: Optional[np.ndarray] = None
        self.age_days: Optional[np.ndarray] = None
        self.variance_scores: Optional[np.ndarray] = None
        self.confidence_scores: Optional[np.ndarray] = None
        self.outlier_mask: Optional[np.ndarray] = None
        self.drift_weights: Optional[np.ndarray] = None

        self.feedback_bias: Dict[str, float] = defaultdict(float)
        self.runtime_config: Dict[str, Any] = {}

        self.candidate_k = 24

    def _to_documents(self, records: Sequence[Dict[str, Any]]) -> List[Document]:
        docs: List[Document] = []
        for row in records:
            doc_id = str(row.get("id", f"doc_{len(docs)}"))
            text = str(row.get("text", "")).strip()
            if not text:
                continue

            modality = str(row.get("modality", "text"))
            if self.method_id in {"10.1_joint_embedding_index", "10.2_cross_modal_linked_index"}:
                text = f"[{modality}] {text}"

            metadata = {k: v for k, v in row.items() if k != "text"}
            metadata["doc_id"] = doc_id
            docs.append(Document(text=text, metadata=metadata))
        return docs

    def _build_structural_nodes(self, docs: Sequence[Document]) -> List[BaseNode]:
        nodes: List[BaseNode] = []
        for doc in docs:
            md = dict(doc.metadata or {})
            for heading, depth, body in chunk_structural(str(doc.text_resource.text)):
                metadata = dict(md)
                metadata["heading"] = heading
                metadata["depth"] = depth
                nodes.append(TextNode(text=body, metadata=metadata))
        return nodes

    def _build_scene_nodes(self, docs: Sequence[Document]) -> List[BaseNode]:
        nodes: List[BaseNode] = []
        for doc in docs:
            md = dict(doc.metadata or {})
            text = str(doc.text_resource.text)
            if str(md.get("modality", "text")).lower() == "video":
                for scene_name, scene_text in extract_scene_chunks(text):
                    metadata = dict(md)
                    metadata["scene"] = scene_name
                    nodes.append(TextNode(text=scene_text, metadata=metadata))
            else:
                nodes.append(TextNode(text=text, metadata=md))
        return nodes

    def _build_nodes(self, docs: Sequence[Document]) -> Tuple[List[BaseNode], List[BaseNode]]:
        method = self.method_id

        if method == "11.1_self_tuning_index":
            lengths = [len(tokenize(str(doc.text_resource.text))) for doc in docs]
            median_len = float(np.median(lengths)) if lengths else 120.0
            if median_len < 120:
                self.runtime_config["chunk_size"] = 80
            elif median_len < 220:
                self.runtime_config["chunk_size"] = 120
            else:
                self.runtime_config["chunk_size"] = 200

        if method == "1.5_small_to_big_hierarchical_chunking":
            parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[512, 128], chunk_overlap=40)
            all_nodes = parser.get_nodes_from_documents(docs)
            leaf_nodes = get_leaf_nodes(all_nodes)
            for node in leaf_nodes:
                rel = (node.relationships or {}).get(NodeRelationship.PARENT)
                if rel is not None:
                    self.parent_map[node.node_id] = rel.node_id
            return all_nodes, leaf_nodes

        if method in {"4.1_tree_structured_index", "4.2_summary_augmented_index", "4.3_structural_aware_index"}:
            nodes = self._build_structural_nodes(docs)
            return nodes, nodes

        if method == "10.3_scene_aware_media_index":
            nodes = self._build_scene_nodes(docs)
            return nodes, nodes

        if method == "1.3_adaptive_semantic_chunking":
            try:
                parser = SemanticSplitterNodeParser.from_defaults(
                    embed_model=Settings.embed_model,
                    breakpoint_percentile_threshold=90,
                )
            except Exception:
                parser = SentenceSplitter.from_defaults(chunk_size=120, chunk_overlap=20)
            nodes = parser.get_nodes_from_documents(docs)
            return nodes, nodes

        if method in {"1.4_query_aware_dynamic_chunking", "1.6_sentence_level_indexing"}:
            window = 2 if method == "1.4_query_aware_dynamic_chunking" else 1
            parser = SentenceWindowNodeParser.from_defaults(window_size=window)
            nodes = parser.get_nodes_from_documents(docs)
            return nodes, nodes

        if method == "1.2_sliding_window_overlap":
            parser = TokenTextSplitter(chunk_size=120, chunk_overlap=40)
            nodes = parser.get_nodes_from_documents(docs)
            return nodes, nodes

        chunk_size = int(self.runtime_config.get("chunk_size", 120))
        parser = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        nodes = parser.get_nodes_from_documents(docs)
        return nodes, nodes

    def _build_sparse_structures(self, texts: Sequence[str]) -> None:
        tokenized = [tokenize(t) for t in texts]
        self.term_freqs = [Counter(tokens) for tokens in tokenized]
        for idx, tokens in enumerate(tokenized):
            for token in set(tokens):
                self.inverted[token].add(idx)

    def _fit_clusters(self, k: int = 6) -> None:
        if self.node_embeddings is None or len(self.node_embeddings) == 0:
            return
        n_samples = len(self.node_embeddings)
        n_clusters = max(2, min(k, n_samples))
        model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        self.cluster_labels = model.fit_predict(self.node_embeddings)
        self.cluster_centers = model.cluster_centers_.astype(np.float32)

    def _fit_semantic_hashing(self, bits: int = 32) -> None:
        if self.node_embeddings is None:
            return
        rng = np.random.default_rng(42)
        self.hash_planes = rng.normal(0.0, 1.0, size=(self.node_embeddings.shape[1], bits)).astype(np.float32)
        projections = np.dot(self.node_embeddings, self.hash_planes)
        self.hash_codes = (projections > 0).astype(np.int8)

    def _summarize_text(self, text: str) -> str:
        sents = split_sentences(text)
        if not sents:
            return text[:160]
        if len(sents) == 1:
            return sents[0]
        return f"{sents[0]} {sents[-1]}"

    def _build_entity_index_graph(self) -> None:
        for idx, node in enumerate(self.index_nodes):
            for entity in extract_entities(node_text(node)):
                key = entity.lower()
                self.entity_index[key].add(idx)

        if self.method_id == "5.1_knowledge_graph_index" and nx is not None:
            graph = nx.Graph()
            for idx, node in enumerate(self.index_nodes):
                cnode = f"node::{idx}"
                graph.add_node(cnode, node_type="chunk")
                ents = extract_entities(node_text(node))
                for ent in ents:
                    enode = f"entity::{ent.lower()}"
                    graph.add_node(enode, node_type="entity")
                    graph.add_edge(cnode, enode)
            self.entity_graph = graph

    def _build_semantic_graph(self, k: int = 4) -> None:
        if self.node_embeddings is None:
            return
        sims = np.dot(self.node_embeddings, self.node_embeddings.T)
        np.fill_diagonal(sims, -np.inf)
        for i in range(len(self.index_nodes)):
            neighbors = np.argsort(-sims[i])[:k]
            for n in neighbors:
                self.semantic_graph[i].add(int(n))
                self.semantic_graph[int(n)].add(i)

    def _build_outlier_mask(self) -> None:
        if self.node_embeddings is None or len(self.node_embeddings) < 8:
            self.outlier_mask = np.ones(len(self.index_nodes), dtype=bool)
            return
        model = IsolationForest(contamination=0.08, random_state=42)
        labels = model.fit_predict(self.node_embeddings)
        self.outlier_mask = labels > 0

    def _build_drift_weights(self) -> None:
        if self.node_embeddings is None:
            return
        center = np.mean(self.node_embeddings, axis=0)
        dists = np.linalg.norm(self.node_embeddings - center, axis=1)
        if np.max(dists) <= 1e-6:
            self.drift_weights = np.ones_like(dists)
            return
        norm = dists / (np.max(dists) + 1e-6)
        self.drift_weights = (1.0 - 0.5 * norm).astype(np.float32)

    def _build_causal_graph(self) -> None:
        markers = ["because", "led to", "resulted in", "therefore", "caused"]
        marker_to_nodes: Dict[str, List[int]] = defaultdict(list)
        for idx, node in enumerate(self.index_nodes):
            text = node_text(node).lower()
            for marker in markers:
                if marker in text:
                    marker_to_nodes[marker].append(idx)
        for node_ids in marker_to_nodes.values():
            for i in node_ids:
                for j in node_ids:
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
            try:
                raw = Settings.embed_model.get_text_embedding_batch(self.summaries)
                self.summary_embeddings = np.asarray(raw, dtype=np.float32)
            except Exception:
                self.summary_embeddings = None

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

    def build(self, documents: Sequence[Dict[str, Any]]) -> None:
        self.documents = [dict(doc) for doc in documents]

        self.embed_model, self.embed_model_v2, self.embed_source = _configure_embed_models()

        llama_docs = self._to_documents(self.documents)
        self.all_nodes, self.index_nodes = self._build_nodes(llama_docs)

        if self.method_id == "1.5_small_to_big_hierarchical_chunking":
            self.storage_context = StorageContext.from_defaults()
            self.storage_context.docstore.add_documents(self.all_nodes)
            self.vector_index = VectorStoreIndex(self.index_nodes, storage_context=self.storage_context)
        else:
            self.vector_index = VectorStoreIndex(self.index_nodes)

        self.dense_retriever = self.vector_index.as_retriever(similarity_top_k=self.candidate_k)

        if BM25Retriever is not None:
            try:
                bm25_top_k = min(self.candidate_k, max(1, len(self.index_nodes)))
                self.bm25_retriever = BM25Retriever.from_defaults(nodes=self.index_nodes, similarity_top_k=bm25_top_k)
            except Exception:
                self.bm25_retriever = None

        self.node_lookup = {node.node_id: node for node in self.all_nodes}
        self.node_id_to_idx = {node.node_id: idx for idx, node in enumerate(self.index_nodes)}

        texts = [node_text(node) for node in self.index_nodes]
        self._build_sparse_structures(texts)

        try:
            raw_embeddings = Settings.embed_model.get_text_embedding_batch(texts)
            self.node_embeddings = np.asarray(raw_embeddings, dtype=np.float32)
        except Exception:
            self.node_embeddings = np.zeros((len(texts), 384), dtype=np.float32)

        if self.method_id == "11.3_continual_re_embedding_index":
            if self.embed_model_v2 is not None:
                try:
                    raw_v2 = self.embed_model_v2.get_text_embedding_batch(texts)
                    self.node_embeddings_v2 = np.asarray(raw_v2, dtype=np.float32)
                except Exception:
                    self.node_embeddings_v2 = None
            if self.node_embeddings_v2 is None:
                self.node_embeddings_v2 = np.roll(self.node_embeddings, shift=1, axis=1)

        self.trust_scores = np.array([
            safe_float((node.metadata or {}).get("trust_score", 0.8), 0.8) for node in self.index_nodes
        ], dtype=np.float32)

        today = date.today()
        age_days: List[float] = []
        for node in self.index_nodes:
            raw_ts = (node.metadata or {}).get("timestamp")
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

        variance_scores = np.array([self._estimate_variance(t) for t in texts], dtype=np.float32)
        self.variance_scores = variance_scores
        self.confidence_scores = np.exp(-2.0 * variance_scores).astype(np.float32)

        self._build_method_specific_structures(texts)

    def _query_embedding(self, query: str, use_v2: bool = False) -> np.ndarray:
        if use_v2 and self.embed_model_v2 is not None:
            try:
                return np.asarray(self.embed_model_v2.get_query_embedding(query), dtype=np.float32)
            except Exception:
                pass
        return np.asarray(Settings.embed_model.get_query_embedding(query), dtype=np.float32)

    def _dense_scores(self, query: str, use_v2: bool = False) -> np.ndarray:
        n = len(self.index_nodes)
        if n == 0:
            return np.array([], dtype=np.float32)

        q = self._query_embedding(query, use_v2=use_v2)
        emb = self.node_embeddings_v2 if (use_v2 and self.node_embeddings_v2 is not None) else self.node_embeddings
        if emb is None:
            return np.zeros(n, dtype=np.float32)

        dot_scores = np.dot(emb, q).astype(np.float32)

        if self.dense_retriever is None:
            return dot_scores

        retriever_scores = np.full(n, -np.inf, dtype=np.float32)
        try:
            hits = self.dense_retriever.retrieve(query)
        except Exception:
            hits = []

        for hit in hits:
            idx = self.node_id_to_idx.get(hit.node.node_id)
            if idx is None:
                continue
            value = safe_float(hit.score, 0.0)
            if not np.isfinite(retriever_scores[idx]) or value > retriever_scores[idx]:
                retriever_scores[idx] = value

        return np.where(np.isfinite(retriever_scores), retriever_scores, dot_scores)

    def _bm25_scores(self, query: str) -> np.ndarray:
        n = len(self.index_nodes)
        if n == 0:
            return np.array([], dtype=np.float32)
        if self.bm25_retriever is None:
            return self._inverted_scores(query)

        scores = np.zeros(n, dtype=np.float32)
        try:
            hits = self.bm25_retriever.retrieve(query)
        except Exception:
            hits = []

        for hit in hits:
            idx = self.node_id_to_idx.get(hit.node.node_id)
            if idx is None:
                continue
            scores[idx] = max(scores[idx], safe_float(hit.score, 0.0))
        return scores

    def _inverted_scores(self, query: str) -> np.ndarray:
        q_tokens = tokenize(query)
        n_docs = len(self.index_nodes)
        scores = np.zeros(n_docs, dtype=np.float32)
        for token in q_tokens:
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
            return np.ones(len(self.index_nodes), dtype=bool)
        q = self._query_embedding(query)
        centroid_scores = np.dot(self.cluster_centers, q)
        keep_clusters = set(np.argsort(-centroid_scores)[:top_clusters].tolist())
        return np.array([int(label) in keep_clusters for label in self.cluster_labels], dtype=bool)

    def _semantic_hash_candidates(self, query: str, threshold: int = 8) -> np.ndarray:
        if self.hash_planes is None or self.hash_codes is None:
            return np.ones(len(self.index_nodes), dtype=bool)
        q = self._query_embedding(query)
        q_code = (np.dot(q, self.hash_planes) > 0).astype(np.int8)
        distances = np.sum(np.abs(self.hash_codes - q_code), axis=1)
        if np.min(distances) > threshold:
            keep = np.argsort(distances)[: min(20, len(distances))]
            mask = np.zeros(len(distances), dtype=bool)
            mask[keep] = True
            return mask
        return distances <= threshold

    def _score_query_aware(self, query: str) -> np.ndarray:
        scores = self._dense_scores(query)
        q_tokens = set(tokenize(query))
        for idx, node in enumerate(self.index_nodes):
            text = node_text(node)
            overlap = len(q_tokens & set(tokenize(text)))
            scores[idx] += 0.2 * overlap
            window_text = str((node.metadata or {}).get("window", ""))
            if window_text:
                window_overlap = len(q_tokens & set(tokenize(window_text)))
                scores[idx] += 0.08 * window_overlap
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

    def _apply_scene_aware_media_adjustment(self, query: str, scores: np.ndarray) -> np.ndarray:
        q_tokens = set(tokenize(query))
        if not q_tokens:
            return scores

        scene_bonus = np.array(
            [
                0.2
                if any(tok in str((node.metadata or {}).get("scene", "")).lower() for tok in q_tokens)
                else 0.0
                for node in self.index_nodes
            ],
            dtype=np.float32,
        )
        return scores + scene_bonus

    def _apply_feedback_bias(self, scores: np.ndarray) -> np.ndarray:
        bias = np.array([self.feedback_bias.get(node.node_id, 0.0) for node in self.index_nodes], dtype=np.float32)
        return scores + bias

    def _apply_continual_reembedding_fusion(self, query: str, scores: np.ndarray) -> np.ndarray:
        v1 = normalize_scores(scores)
        v2 = normalize_scores(self._dense_scores(query, use_v2=True))
        return 0.35 * v1 + 0.65 * v2

    def _apply_causal_graph_boost(self, query: str, scores: np.ndarray) -> np.ndarray:
        query_lower = query.lower()
        if not any(marker in query_lower for marker in self._CAUSAL_QUERY_MARKERS):
            return scores

        boosted = scores.copy()
        for seed in np.argsort(-scores)[: min(5, len(scores))]:
            for nbr in self.causal_graph.get(int(seed), set()):
                boosted[nbr] += 0.2
        return boosted

    def _apply_explanation_bonus(self, query: str, scores: np.ndarray) -> np.ndarray:
        q_lower = query.lower()
        if not any(token in q_lower for token in self._EXPLANATION_QUERY_MARKERS):
            return scores

        bonus = np.array(
            [
                0.15 if any(marker in node_text(node).lower() for marker in self._EXPLANATION_TEXT_MARKERS) else 0.0
                for node in self.index_nodes
            ],
            dtype=np.float32,
        )
        return scores + bonus

    def _apply_plan_bonus(self, query: str, scores: np.ndarray) -> np.ndarray:
        q_lower = query.lower()
        if not any(token in q_lower for token in self._PLAN_QUERY_MARKERS):
            return scores

        bonus = np.array(
            [
                0.18 if any(marker in node_text(node).lower() for marker in self._PLAN_TEXT_MARKERS) else 0.0
                for node in self.index_nodes
            ],
            dtype=np.float32,
        )
        return scores + bonus

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
            q = self._query_embedding(query).astype(np.float16)
            emb = self.node_embeddings.astype(np.float16) if self.node_embeddings is not None else None
            if emb is not None:
                scores = np.dot(emb, q).astype(np.float32)

        elif m == "2.4_hybrid_dense_partition_index":
            mask = self._cluster_candidate_mask(query, top_clusters=3)
            masked_scores = np.where(mask, scores, -np.inf)
            cluster_prior = np.zeros_like(scores)
            if self.cluster_labels is not None and self.cluster_centers is not None:
                q = self._query_embedding(query)
                centroid_scores = np.dot(self.cluster_centers, q)
                norm_centroids = normalize_scores(centroid_scores)
                for i, label in enumerate(self.cluster_labels):
                    cluster_prior[i] = norm_centroids[int(label)]
            scores = 0.7 * normalize_scores(masked_scores) + 0.3 * cluster_prior

        elif m == "4.1_tree_structured_index":
            depth_boost = np.array([
                1.0 / (1.0 + safe_float((node.metadata or {}).get("depth", 0), 0.0))
                for node in self.index_nodes
            ], dtype=np.float32)
            scores = scores * (0.7 + 0.3 * depth_boost)

        elif m == "4.2_summary_augmented_index":
            if self.summary_embeddings is not None:
                q = self._query_embedding(query)
                summary_scores = np.dot(self.summary_embeddings, q)
                scores = 0.4 * normalize_scores(scores) + 0.6 * normalize_scores(summary_scores)

        elif m == "4.3_structural_aware_index":
            q_tokens = set(tokenize(query))
            bonuses = []
            for node in self.index_nodes:
                heading = str((node.metadata or {}).get("heading", "")).lower()
                bonus = sum(0.15 for token in q_tokens if token in heading)
                bonuses.append(bonus)
            scores = scores + np.array(bonuses, dtype=np.float32)

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

        elif m == "7.3_real_time_streaming_index":
            if self.age_days is not None:
                recency = normalize_scores(-self.age_days)
                scores = 0.75 * normalize_scores(scores) + 0.25 * recency

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

        elif m == "10.3_scene_aware_media_index":
            scores = self._apply_scene_aware_media_adjustment(query, scores)

        elif m == "11.2_feedback_driven_index":
            scores = self._apply_feedback_bias(scores)

        elif m == "11.3_continual_re_embedding_index":
            scores = self._apply_continual_reembedding_fusion(query, scores)

        elif m == "12.1_causal_graph_index":
            scores = self._apply_causal_graph_boost(query, scores)

        elif m == "12.2_explanation_aware_index":
            scores = self._apply_explanation_bonus(query, scores)

        elif m == "12.3_plan_oriented_index":
            scores = self._apply_plan_bonus(query, scores)

        return scores

    def search(self, query: str, top_k: int = 5) -> List[ScoredNode]:
        if not self.index_nodes:
            return []

        scores = self._scores_for_method(query)
        valid = np.isfinite(scores)
        if not valid.any():
            return []

        order = np.argsort(-scores)
        raw_rows: List[ScoredNode] = []
        for idx in order:
            idx = int(idx)
            if not np.isfinite(scores[idx]):
                continue
            node = self.index_nodes[idx]
            meta = dict(node.metadata or {})
            doc_id = str(meta.get("doc_id") or meta.get("id") or node.node_id)
            raw_rows.append(
                ScoredNode(
                    node_id=node.node_id,
                    doc_id=doc_id,
                    score=float(scores[idx]),
                    text=node_text(node),
                    metadata=meta,
                )
            )
            if len(raw_rows) >= max(top_k * 4, 16):
                break

        if self.method_id == "1.5_small_to_big_hierarchical_chunking":
            merged: Dict[str, ScoredNode] = {}
            for row in raw_rows:
                parent_id = self.parent_map.get(row.node_id)
                if not parent_id:
                    continue
                parent_node = self.node_lookup.get(parent_id)
                if parent_node is None:
                    continue
                parent_meta = dict(parent_node.metadata or {})
                parent_doc_id = str(parent_meta.get("doc_id") or parent_meta.get("id") or parent_id)
                existing = merged.get(parent_id)
                candidate = ScoredNode(
                    node_id=parent_id,
                    doc_id=parent_doc_id,
                    score=row.score,
                    text=node_text(parent_node),
                    metadata=parent_meta,
                )
                if existing is None or candidate.score > existing.score:
                    merged[parent_id] = candidate
            raw_rows = sorted(merged.values(), key=lambda x: x.score, reverse=True)

        if self.method_id == "10.2_cross_modal_linked_index":
            by_asset: Dict[str, ScoredNode] = {}
            for row in raw_rows:
                asset_id = str(row.metadata.get("asset_id") or "")
                if not asset_id:
                    continue
                best = by_asset.get(asset_id)
                if best is None or row.score > best.score:
                    by_asset[asset_id] = row

            expanded = list(raw_rows)
            seen_node_ids = {row.node_id for row in raw_rows}
            for asset_id, best_row in by_asset.items():
                for node in self.index_nodes:
                    meta = dict(node.metadata or {})
                    if str(meta.get("asset_id") or "") != asset_id:
                        continue
                    if node.node_id in seen_node_ids:
                        continue
                    expanded.append(
                        ScoredNode(
                            node_id=node.node_id,
                            doc_id=str(meta.get("doc_id") or meta.get("id") or node.node_id),
                            score=float(best_row.score * 0.92),
                            text=node_text(node),
                            metadata=meta,
                        )
                    )
                    seen_node_ids.add(node.node_id)
            raw_rows = sorted(expanded, key=lambda x: x.score, reverse=True)

        final_rows: List[ScoredNode] = []
        seen_docs = set()
        for row in raw_rows:
            if row.doc_id in seen_docs:
                continue
            seen_docs.add(row.doc_id)
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
            ranked_ids = [row.doc_id for row in results]

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
                    if row.doc_id in relevant:
                        self.feedback_bias[row.node_id] += 0.05

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
            "embedding_source": self.embed_source,
            "documents": len(self.documents),
            "nodes": len(self.index_nodes),
            "metrics": metrics,
            "runtime_config": self.runtime_config,
            "per_query": per_query,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


# Backward-compatible alias for previous API name.
IndexingMethodRunner = LlamaIndexMethodRunner


def run_method(
    method_id: str,
    corpus_path: Optional[str | Path] = None,
    queries_path: Optional[str | Path] = None,
    output_dir: str | Path = "results",
    top_k: int = 5,
) -> Dict[str, Any]:
    runner = LlamaIndexMethodRunner(method_id)
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
    parser = argparse.ArgumentParser(description="Run LlamaIndex indexing method experiments.")
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
    print(
        json.dumps(
            {
                "method_id": result["method_id"],
                "embedding_source": result.get("embedding_source"),
                "metrics": result["metrics"],
                "result_path": result["result_path"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    _main()
