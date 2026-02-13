Below is a structured 2026-level taxonomy written in the same academic style and hierarchy as your example.

⸻

A. Indexing

Indexing is the process of organizing raw data into structured representations that enable efficient, accurate, and contextually coherent retrieval. In modern retrieval systems, indexing extends beyond simple chunking to incorporate semantic, structural, temporal, and relational representations. Indexing in 2026 faces several systemic challenges: 1) Incomplete semantic coverage due to segmentation artifacts, 2) Retrieval noise amplification under scale, 3) Loss of provenance and reasoning traceability, 4) Distributional drift and adversarial manipulation, and 5) Multi-modal heterogeneity across data sources.

Indexing methods can be categorized as follows.

⸻

1) Chunk-Based Indexing

Chunk-based indexing partitions documents into manageable units. The design of chunk boundaries significantly influences retrieval precision and synthesis coherence.

1.1 Fixed-Length Chunking

Documents are divided into uniform token windows.
Advantages include implementation simplicity and predictable memory usage.
Limitations include semantic boundary violations and context fragmentation.

1.2 Sliding Window with Overlap

Overlapping chunks preserve transitional context.
Let di denote chunk i with size Li = |di| and overlap Loi = |di ∩ di+1|.
While overlap improves continuity, it increases storage redundancy and noise.

1.3 Adaptive Semantic Chunking

Chunks are defined based on semantic coherence rather than fixed length.
Segmentation uses discourse markers, topic shifts, or embedding similarity thresholds.

1.4 Query-Aware Dynamic Chunking

Chunk boundaries are adjusted at query time based on intent or entity focus.
This reduces unnecessary context injection and improves precision.

1.5 Small-to-Big Hierarchical Chunking

Smaller chunks are indexed for retrieval, while larger parent chunks are used for synthesis.
This separates retrieval granularity from generation context size.

1.6 Sentence-Level Indexing

Individual sentences are indexed independently and linked to surrounding context windows.

⸻

2) Vector-Based Indexing

Vector indexing encodes text into dense embeddings and enables similarity-based retrieval.

2.1 Flat Dense Index

All embeddings stored in contiguous space; retrieval via exhaustive search.
High recall but computationally expensive at scale.

2.2 Approximate Nearest Neighbor Index

Uses graph or partition-based structures such as HNSW or IVF to accelerate similarity search.
Balances latency and recall.

2.3 Quantized Vector Index

Applies compression techniques such as product quantization to reduce memory footprint.

2.4 Hybrid Dense Partition Index

Combines clustering and graph-based traversal for scalable retrieval.

⸻

3) Sparse and Lexical Indexing

Sparse indexing captures token-level information.

3.1 Inverted Index

Maps terms to posting lists of document identifiers.
Enables fast keyword search and boolean filtering.

3.2 Sparse-Dense Fusion Index

Combines sparse token matching with dense embedding similarity.
Improves lexical grounding while preserving semantic recall.

3.3 Weighted Term Index

Uses statistical weighting such as TF-IDF or BM25 for ranking relevance.

⸻

4) Hierarchical Indexing

Hierarchical indexing organizes documents into multi-level structures.

4.1 Tree-Structured Index

Documents structured as nodes in parent-child hierarchies.
Each node stores summaries and pointers to children.

4.2 Summary-Augmented Index

Each node stores compressed semantic summaries to guide traversal.

4.3 Structural-Aware Index

Chunking aligned with document sections, headings, tables, or semantic blocks.

⸻

5) Graph-Based Indexing

Graph indexing models relational structure between entities and passages.

5.1 Knowledge Graph Index

Corpus represented as graph G = {V, E, X}, where nodes V represent entities or passages, edges E represent semantic or structural relations, and node features X encode textual content.

5.2 Entity-Centric Index

Entities are first-class index elements; documents linked via entity mentions.

5.3 Semantic Similarity Graph

Edges constructed based on embedding similarity thresholds.

5.4 Multi-Hop Retrieval Graph

Supports reasoning chains across connected nodes.

⸻

6) Topic and Cluster-Based Indexing

Documents grouped by latent themes.

6.1 Topic Partition Index

Corpus partitioned into thematic clusters; queries routed to relevant clusters.

6.2 Semantic Hashing Index

Documents mapped to compact binary codes for sublinear search.

⸻

7) Temporal and Streaming Indexing

Designed for dynamic and evolving corpora.

7.1 Time-Decayed Index

Weights assigned based on recency.

7.2 Sliding Temporal Window Index

Maintains separate indices for recent and historical data.

7.3 Real-Time Streaming Index

Supports continuous insertion and incremental updates without full rebuild.

⸻

8) Robust and Adversarial-Aware Indexing

Designed for resilience against noise and manipulation.

8.1 Trust-Weighted Index

Each entry assigned provenance or confidence scores influencing ranking.

8.2 Outlier-Aware Vector Index

Detects anomalous embeddings during insertion.

8.3 Drift-Aware Index

Monitors embedding distribution shifts and triggers re-indexing.

⸻

9) Probabilistic and Uncertainty-Aware Indexing

Incorporates uncertainty into representation.

9.1 Distributional Embedding Index

Stores mean and variance per vector.

9.2 Confidence-Propagating Index

Retrieval scores adjusted by uncertainty estimates.

⸻

10) Multi-Modal Indexing

Handles heterogeneous data types.

10.1 Joint Embedding Index

Maps text, image, audio, and video into shared semantic space.

10.2 Cross-Modal Linked Index

Maintains modality-specific embeddings connected via alignment edges.

10.3 Scene-Aware Media Index

Breaks video into scenes and attaches OCR and transcript metadata.

⸻

11) Adaptive and Continual Indexing

Designed for evolving usage patterns.

11.1 Self-Tuning Index

Automatically adjusts chunk sizes, thresholds, and pruning strategies.

11.2 Feedback-Driven Index

Learns from retrieval interactions to refine ranking and segmentation.

11.3 Continual Re-Embedding Index

Periodically updates embeddings to reflect model upgrades.

⸻

12) Reasoning-Ready Indexing

Supports structured inference rather than flat retrieval.

12.1 Causal Graph Index

Encodes cause-effect relationships.

12.2 Explanation-Aware Index

Associates passages with reasoning chains or evidence annotations.

12.3 Plan-Oriented Index

Annotates content fragments with procedural or action-oriented metadata.

⸻

This taxonomy reflects the full landscape of indexing methodologies deployed or researched in 2026, spanning chunking strategies, vector structures, graph representations, temporal adaptation, robustness, uncertainty modeling, and reasoning-oriented architectures.