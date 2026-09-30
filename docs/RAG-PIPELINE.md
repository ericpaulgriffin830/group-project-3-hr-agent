# RAG pipeline

The seven-component diagram in [`ARCHITECTURE.md`](ARCHITECTURE.md) collapses the
whole `app/rag/` package into three boxes (`retrieve.py`, `answer.py`, the Chroma
index). This is the zoomed-in version: every file in `app/rag/`, plus `ingest.py`
and `scripts/build_index.py`, and which of two paths a piece of data takes —
**built once at deploy** versus **run on every question**.

---

## Build time — once per deploy, not per question

`scripts/build_index.py` runs this at the start of every Render build, because the
free tier has no persistent disk: `data/chroma` is gitignored and rebuilt from the
corpus every time, never committed as data.

```mermaid
graph LR
    CORPUS[("corpus/*.md .html .txt<br/>12 policy documents")]
    ING["<b>ingest.py</b><br/>load_corpus()<br/><i>parses headings into<br/>RawSection / ParsedDocument</i>"]
    CHK["<b>chunk.py</b><br/>chunk_corpus()<br/><i>one section = one chunk,<br/>always — 106 chunks total</i>"]
    EMB["<b>embed.py</b><br/>embed_documents()<br/><i>fastembed ·<br/>BAAI/bge-small-en-v1.5</i>"]
    STO["<b>store.py</b><br/>upsert_chunks()"]
    IDX[("Chroma index<br/>data/chroma<br/>(gitignored, rebuilt every deploy)")]

    CORPUS --> ING --> CHK --> EMB --> STO --> IDX

    classDef rob fill:#dcfce7,stroke:#15803d,color:#14532d
    classDef ext fill:#f3f4f6,stroke:#6b7280,color:#374151
    class ING,CHK,EMB,STO rob
    class CORPUS,IDX ext
```

`chunk.py` never talks to `embed.py` or `store.py` directly — `build_index.py` is
the only place all four modules meet, which is what keeps each one testable with
plain Python objects and no live index.

---

## Query time — one `search_policy_documents` call

This is what runs inside `retrieve()` for every RAG-backed tool call the agent
makes. Two legs run over the same 106 chunks and are fused before anything is
returned.

```mermaid
graph TB
    Q["question text<br/><i>from search_policy_documents(query, k)</i>"]

    subgraph vector["vector leg"]
        VQ["embed.py<br/>embed_query()"]
        VS["store.py<br/>vector_search()"]
        CH[("Chroma index")]
        VQ --> VS --> CH
    end

    subgraph keyword["BM25 leg"]
        BM["retrieve.py<br/>_bm25_ranked_ids()<br/><i>rank_bm25, in-process,<br/>built from chunk_corpus()</i>"]
    end

    Q --> VQ
    Q --> BM

    GATE{"relevance gate<br/>vector ≥ 0.6<br/>OR any BM25 hit?"}
    CH --> GATE
    BM --> GATE

    NOEVID["chunks: []<br/><i>no genuine match on either leg</i>"]
    GATE -->|no| NOEVID

    RRF["_rrf_fuse()<br/><i>weighted RRF — 0.8 vector / 0.2 BM25,<br/>rank-based, not raw score</i>"]
    GATE -->|yes| RRF

    CAP["_cap_per_document()<br/><i>≤ 2 chunks per doc_id</i>"]
    RRF --> CAP

    OUT["PolicyChunk[]<br/><i>doc_id · title · section · snippet · score</i>"]
    CAP --> OUT

    classDef rob fill:#dcfce7,stroke:#15803d,color:#14532d
    classDef ext fill:#f3f4f6,stroke:#6b7280,color:#374151
    classDef gate fill:#fee2e2,stroke:#b91c1c,color:#7f1d1d
    class VQ,VS,BM,RRF,CAP rob
    class Q,OUT,CH ext
    class GATE,NOEVID gate
```

`get_policy_section(doc_id, section_id)` is a separate, simpler path: it looks up
the full section text directly from `ingest.py`'s parsed sections
(`retrieve.py`'s `_section_lookup()`), never from a chunk — the whole reason
`get_policy_section` exists is to return the section even if chunking had ever
split it, though as of the current chunking strategy a chunk already *is* the
full section.

---

## From retrieved chunks to a cited answer

`search_policy_documents`'s output above is only evidence. `answer.py`'s
`synthesize()` is the seam (Contract C) that turns evidence into an answer with
real citations:

```mermaid
graph LR
    EV["evidence[]<br/><i>from search_policy_documents<br/>+ check_policy_compliance's policy_refs</i>"]
    TR["tool_results[]<br/><i>employee-specific data,<br/>workflow mode only</i>"]

    ENR["_enrich()<br/><i>backfills a sparse {doc_id, section}<br/>ref via get_section(); drops it<br/>if it can't be resolved</i>"]
    PROMPT["prompt + system instructions<br/><i>≤ 10 passages shown</i>"]
    LLM["app/llm.py<br/>complete()<br/><i>Groq, temp 0, fixed seed</i>"]
    PARSE["_parse_completion()<br/><i>reads the model's own META<br/>trailer: cited_doc_ids,<br/>unsupported_flags</i>"]
    XREF["cross-reference cited_doc_ids<br/>against evidence actually shown<br/><i>a doc_id the model invents<br/>cannot match anything</i>"]
    OUT2["answer, citations,<br/>basis, confidence"]

    EV --> ENR --> PROMPT
    TR --> PROMPT
    PROMPT --> LLM --> PARSE --> XREF --> OUT2

    classDef rob fill:#dcfce7,stroke:#15803d,color:#14532d
    classDef ext fill:#f3f4f6,stroke:#6b7280,color:#374151
    class ENR,PROMPT,PARSE,XREF rob
    class EV,TR,LLM,OUT2 ext
```

The model never gets to declare its own citations directly — `synthesize()`
builds the `citations` field itself by matching `cited_doc_ids` back against the
evidence dicts that were actually in the prompt, so a hallucinated doc_id is
silently unmatched rather than trusted.
