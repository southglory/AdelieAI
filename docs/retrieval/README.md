# Retrieval

## 책임

지식 / 사실 / 문맥 청크의 *검색*. 답변 생성 (= [`personas/`](../personas/)) 과 분리.

이 영역은:
- 비구조화 텍스트 RAG (벡터 + BM25 + RRF + reranker) — T2
- KG / 그래프 검색 (RDF + SPARQL + OWL reasoner) — T4
- 하이브리드 (vector + KG)

## 핵심 파일

### 벡터 RAG (T2)

- [`core/retrieval/protocols.py`](../../core/retrieval/protocols.py) — `Embedder`, `Reranker`, `VectorStore`, `BM25Index`, `Chunker`, `Retriever`
- [`core/retrieval/embedder.py`](../../core/retrieval/embedder.py) — TransformersEmbedder (multilingual-e5-small)
- [`core/retrieval/bm25.py`](../../core/retrieval/bm25.py) — InMemoryBM25
- [`core/retrieval/vectorstore.py`](../../core/retrieval/vectorstore.py) — ChromaVectorStore
- [`core/retrieval/reranker.py`](../../core/retrieval/reranker.py) — CrossEncoderReranker (bge-reranker-v2-m3)
- [`core/retrieval/hybrid.py`](../../core/retrieval/hybrid.py) — HybridRetriever (BM25 + dense + RRF + rerank)
- [`core/retrieval/chunker.py`](../../core/retrieval/chunker.py) — RecursiveTextSplitter
- [`core/retrieval/ingest.py`](../../core/retrieval/ingest.py) — IngestService

### KG 검색 (T4)

- [`core/retrieval/graph_retriever.py`](../../core/retrieval/graph_retriever.py) — `GraphRetriever`, `OWLReasoner`, `Triple`, `GraphHit` Protocol
- [`core/retrieval/graph_retriever_stub.py`](../../core/retrieval/graph_retriever_stub.py) — Stub (의존성 0)
- [`core/retrieval/graph_retriever_rdflib.py`](../../core/retrieval/graph_retriever_rdflib.py) — 실구현 (rdflib + owlrl)
- [`core/retrieval/dragon_lore.ttl`](../../core/retrieval/dragon_lore.ttl) — mock 도메인 KG (Turtle 형식)

## 현재 상태 (v0.2.5)

### Vector RAG (T2)
- ✅ multilingual-e5 임베딩
- ✅ ChromaDB 벡터 스토어
- ✅ BM25 + dense 하이브리드 + RRF fusion
- ✅ bge-reranker-v2-m3 cross-encoder rerank
- ✅ Recursive splitter chunking

### KG-RAG (T4)
- ✅ rdflib 진짜 SPARQL (`descendantOf+` transitive 등)
- ✅ owlrl OWL-RL forward chaining
- ✅ Stub fallback (의존성 없을 때)
- 🔄 사용자 도메인 KG 임포트 (현재 dragon_lore 만 hardcoded)
- ❌ Networked triple store (Fuseki / GraphDB / Stardog)
- ❌ Vector + KG 하이브리드 fusion

## 사용법

### Vector RAG
```python
from core.retrieval.hybrid import HybridRetriever
retriever = HybridRetriever(
    embedder=embedder,
    vector_store=vector_store,
    bm25=bm25,
    reranker=reranker,
)
hits = await retriever.search(query="...", top_k=4)
```

### KG-RAG
```python
from core.retrieval.graph_retriever_rdflib import RdflibGraphRetriever
g = RdflibGraphRetriever()  # 자동으로 dragon_lore.ttl 파싱
hits = g.query("PREFIX adel: <http://adelie.local/lore#> SELECT ?o WHERE { adel:Self adel:descendantOf+ ?o }")
neighborhood = g.expand("Erebor", depth=2)
```

## 평가

[`docs/eval/`](../eval/) 의 RAG 메서드:
- [Faithfulness](../eval/methods/faithfulness.md) — 답변이 검색 결과만 인용
- [Answer Relevance](../eval/methods/answer_relevance.md) — 답변이 질문에 답함
- [Citation Coverage](../eval/methods/citation_coverage.md) — 검색 결과가 답변에 활용됨

세션 (LangGraph) 단위로 자동 측정. 채팅 단위 평가는 Step 6.1.

## 로드맵

- [ ] **vector + KG 하이브리드** — 같은 query 가 두 retriever 에 분기, 결과 fusion
- [ ] **Fuseki / GraphDB 통합** — 네트워크 triple store 지원
- [ ] **Custom domain KG 로더** — 사용자가 자기 `.ttl` 파일 mount → 자동 적용
- [ ] **Reranker 도 KG 결과에 적용** — 그래프 hit 의 LLM-judge 재정렬

## 함정

- **`top_k` 너무 ↑면 LLM 컨텍스트 오염** — top_k=4 정도가 sweet spot
- **Reranker 가 latency 의 dominant cost** — bge-reranker-v2-m3 는 4 청크 rerank 에 0.5-1 초
- **rdflib SPARQL 의 property path 와 owlrl 추론은 다른 매커니즘** — `descendantOf+` 는 SPARQL 가 transitive 처리, 클래스 멤버십 추론은 owlrl 가 처리. 둘 다 함께 사용하면 답이 뒤섞일 수 있음.
- **KG triple 형식이 RDF/turtle 그대로 LLM 에 가면 무시됨** — [`personas/grounding.py`](../../core/personas/grounding.py) 의 `_render_fact` 가 자연 한국어로 변환 필수.

## 기여 가이드

새 retriever 추가:
1. `core/retrieval/{name}.py` — Protocol 적합
2. `tests/test_{name}.py` — Protocol 적합 + 기본 동작
3. `build_app` 디스패치 (의존성 있으면 실구현, 없으면 stub fallback)
4. PR commit prefix: `feat(retrieval):`
