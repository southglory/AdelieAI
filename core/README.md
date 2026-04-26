# `core/` — differentia-llm 엔진 설계 (D-5)

**작성**: 2026-04-24 (D-5 · 이테크시스템 기술 면접 준비)
**범위**: 엔진 모듈 경계 · Protocol 인터페이스 · 데이터 모델 · D-4~D-3 POC 스코프
**상태**: 설계 단계 — 실제 코드는 D-4부터

---

## 0. 엔진 전체 지도

```
core/
├── README.md                # 이 문서
├── schemas/                 # 공유 Pydantic 데이터 모델
├── retrieval/               # RAG 파이프라인 (chunking · embedding · vector store · hybrid · reranker)
├── agent/                   # LangGraph 기반 multi-agent 오케스트레이션
├── session/                 # Agent 세션 상태 머신 + 이벤트 소싱 (LLM Ops Dashboard 엔진)
├── eval/                    # RAGAS 기반 평가
├── serving/                 # LLM client 추상화 (OpenAI / vLLM / Anthropic)
├── api/                     # FastAPI 최소 엔드포인트 (Swagger 시연용)
└── training/                # SFT + DPO — 30일 트랙 Week 2~4 (D-5 범위 외)
```

`differentiations/` 에이전트(scholar·founder·brain)는 이 `core/`를 import해서 자기 use case를 붙인다.
D-5 문서는 `core/` 경계만 다룸. `differentiations/`는 해당 에이전트 책임.

---

## 1. 설계 원칙 (4)

1. **Protocol-first** — 모든 모듈은 `protocols.py`의 Protocol 클래스로 계약 정의. 구현체는 교체 가능.
2. **엔티티 치환 용이** — `session/` 데이터 모델은 **elice_cloud VM 상태 머신 패턴을 LLM Agent 세션으로 엔티티만 치환**한 구조. 면접 내러티브 "같은 엔진, 다른 엔티티"의 실증.
3. **POC 간편 · 프로덕션 교체 쉽게** — POC는 ChromaDB + SQLite, 프로덕션 swap은 URL/config 한 줄 (Milvus·pgvector·Postgres).
4. **사이드이펙트 분리** — 외부 의존(LLM API · Vector DB · DB)은 Protocol 경계에서만. 테스트 시 MockStore로 치환.

---

## 2. 모듈별 책임 · 인터페이스

### 2.1 `schemas/` — 공유 데이터 모델

```python
# schemas/agent.py
from enum import Enum
from datetime import datetime
from typing import Literal
from pydantic import BaseModel

class SessionState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AgentSession(BaseModel):
    id: str                           # UUID
    user_id: str
    goal: str                         # 사용자 질의 원문
    state: SessionState
    model_spec: str                   # e.g. "claude-opus-4-7", "qwen2.5-7b"
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None = None

class AgentEvent(BaseModel):
    id: str
    session_id: str
    event_type: Literal["state_transition", "tool_call", "retrieval", "llm_call", "error", "final"]
    from_state: SessionState | None = None
    to_state: SessionState | None = None
    payload: dict                     # tool args·retrieved docs·LLM prompt·error trace
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    occurred_at: datetime

# schemas/retrieval.py
class Chunk(BaseModel):
    id: str
    doc_id: str
    text: str
    metadata: dict                    # source·page·section·effective_from (리니지)
    embedding: list[float] | None = None

class RetrievedContext(BaseModel):
    chunks: list[Chunk]
    scores: list[float]
    method: Literal["bm25", "dense", "hybrid", "reranked"]

# schemas/eval.py
class EvalResult(BaseModel):
    session_id: str
    faithfulness: float               # 0~1
    answer_relevance: float
    context_precision: float
    context_recall: float | None = None
    notes: str | None = None
```

---

### 2.2 `retrieval/` — RAG 파이프라인

```python
# retrieval/protocols.py
from typing import Protocol
from core.schemas.retrieval import Chunk, RetrievedContext

class Chunker(Protocol):
    def chunk(self, text: str, metadata: dict) -> list[Chunk]: ...

class Embedder(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

class VectorStore(Protocol):
    async def upsert(self, chunks: list[Chunk]) -> None: ...
    async def search_dense(self, query_vec: list[float], k: int, filters: dict | None = None) -> list[tuple[Chunk, float]]: ...

class BM25Index(Protocol):
    async def search_bm25(self, query: str, k: int, filters: dict | None = None) -> list[tuple[Chunk, float]]: ...

class Reranker(Protocol):
    async def rerank(self, query: str, chunks: list[Chunk], top_k: int) -> list[tuple[Chunk, float]]: ...

class Retriever(Protocol):
    """End-to-end: query → hybrid search → rerank → top_k."""
    async def retrieve(self, query: str, k: int = 5, filters: dict | None = None) -> RetrievedContext: ...
```

**POC 구현 선택**
- `chunking.py` — `SemanticChunker` (naïve sentence-based + sliding window fallback)
- `embedding.py` — OpenAI `text-embedding-3-small` (fast, 1536-d)
- `vectorstore.py` — **ChromaDB** (embedded, POC) / **Milvus adapter stub** (production swap)
- `bm25.py` — `rank_bm25` 라이브러리
- `reranker.py` — `cross-encoder/bge-reranker-v2-m3` (HuggingFace)
- `hybrid.py` — RRF (Reciprocal Rank Fusion) dense + BM25
- `pipeline.py` — `HybridRetriever` wire-up

---

### 2.3 `agent/` — LangGraph 멀티 에이전트

```python
# agent/protocols.py
from typing import Protocol, Any

class Tool(Protocol):
    name: str
    description: str
    async def invoke(self, **kwargs) -> Any: ...

class AgentNode(Protocol):
    """LangGraph node signature: state → state (partial)."""
    async def __call__(self, state: dict) -> dict: ...
```

**LangGraph StateGraph 노드 (POC)**
- `planner` — 사용자 질의 분해 + tool 선택 계획
- `retriever_node` — `Retriever` 호출 → `RetrievedContext`
- `sql_tool` — ERP 정형 데이터 조회 (`SQL` 템플릿, 샘플 DB)
- `reasoner` — LLM 추론 (근거·결론·리스크 등급)
- `reporter` — 최종 응답 + 리니지 부착

**상태**: `AgentSession` + `AgentEvent` 누적. 각 노드 실행 후 `session/` 모듈에 이벤트 append.

---

### 2.4 `session/` — Agent 세션 상태 머신 (LLM Ops Dashboard 엔진)

**⭐ 내러티브 하이라이트**: elice_cloud `vm_state_machine.py` 5중 방어를 Agent 세션으로 엔티티 치환.

```python
# session/protocols.py
from typing import Protocol
from core.schemas.agent import AgentSession, AgentEvent, SessionState

class SessionStore(Protocol):
    async def create(self, user_id: str, goal: str, model_spec: str) -> AgentSession: ...
    async def get(self, session_id: str, user_id: str) -> AgentSession | None: ...  # IDOR 방어: user_id 조건
    async def list(self, user_id: str, limit: int = 50) -> list[AgentSession]: ...
    async def transition(self, session_id: str, user_id: str, to: SessionState) -> AgentSession: ...
    async def append_event(self, event: AgentEvent) -> None: ...
    async def events(self, session_id: str, user_id: str) -> list[AgentEvent]: ...  # 이벤트 리플레이
    async def soft_delete(self, session_id: str, user_id: str) -> None: ...  # state → CANCELLED, 이력 보존
```

**상태 머신 규칙** (elice_cloud 패턴 그대로)
- 합법 전이: `PENDING → RUNNING`, `RUNNING → COMPLETED`, `RUNNING → FAILED`, `* → CANCELLED` (END 상태 제외)
- 불법 전이는 DB CHECK로 차단 (`Postgres CHECK` 제약, POC는 Python assert)
- 동시성: row lock + dedup unique index (POC는 in-process lock)

**POC 구현**
- `store.py` — `SQLiteSessionStore` (POC) / `PostgresSessionStore` (stub, 프로덕션 swap)
- `state_machine.py` — 전이 검증 + 이벤트 append 원자적 실행
- `events.py` — 이벤트 append-only + 기간 리플레이

---

### 2.5 `eval/` — RAGAS 기반 평가

```python
# eval/protocols.py
from typing import Protocol
from core.schemas.eval import EvalResult

class Evaluator(Protocol):
    async def evaluate(
        self,
        session_id: str,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> EvalResult: ...
```

**POC 구현**
- `ragas_adapter.py` — `ragas.metrics`의 `faithfulness`, `answer_relevance`, `context_precision` (ground_truth 있을 때 `context_recall`도)
- `report.py` — 세션별 평가 누적 + CSV/JSON export

---

### 2.6 `serving/` — LLM client 추상화

```python
# serving/protocols.py
from typing import Protocol, AsyncIterator

class LLMClient(Protocol):
    async def generate(self, prompt: str, **kwargs) -> str: ...
    async def stream(self, prompt: str, **kwargs) -> AsyncIterator[str]: ...
```

**POC 구현**
- `anthropic_client.py` — 주 모델 (Claude Opus/Sonnet)
- `openai_client.py` — 대안
- `vllm_client.py` — **stub** (OpenAI-compatible REST, 프로덕션 Private LLM 배포용 자리만 만들어둠 — 이테크시스템 JD 대응)

면접 포인트: "vLLM 클라이언트 추상화는 해뒀습니다 — on-prem 배포 시 OpenAI-compatible 엔드포인트만 바꾸면 됩니다."

---

### 2.7 `api/` — FastAPI 최소 엔드포인트 (Swagger 시연)

**설계 원칙**: 엘리스 미니프로젝트 `POST /api/v1/vms`와 **동일한 URL 패턴 구조**로 맞춤 → 면접 시 "같은 엔진 다른 엔티티" 직관 증명.

```
POST   /api/v1/agents/sessions                # 세션 생성 (PENDING)
GET    /api/v1/agents/sessions                # 본인 세션 목록
GET    /api/v1/agents/sessions/{id}           # 단건 + 현재 상태
POST   /api/v1/agents/sessions/{id}/run       # PENDING → RUNNING + LangGraph 실행
GET    /api/v1/agents/sessions/{id}/events    # 이벤트 리플레이
DELETE /api/v1/agents/sessions/{id}           # soft delete → CANCELLED, 이력 보존

POST   /api/v1/rag/query                      # 리트리버 단독 테스트 (디버깅용)
POST   /api/v1/eval/run                       # 세션별 RAGAS 평가

GET    /health
GET    /                                      # 버전·모듈 상태
```

Swagger가 자동 생성됨 → 면접에서 `http://localhost:8000/docs` 열어 바로 시연.

---

## 3. 의존성 (Python 패키지)

```toml
# pyproject.toml 초안 (D-4 첫 작업)
[project]
dependencies = [
    "fastapi>=0.115",
    "pydantic>=2.9",
    "langgraph>=0.2",
    "langchain-core>=0.3",
    "anthropic>=0.40",
    "openai>=1.50",
    "chromadb>=0.5",
    "rank-bm25>=0.2",
    "sentence-transformers>=3.0",       # reranker
    "ragas>=0.2",
    "aiosqlite>=0.20",                  # POC session store
    "sqlalchemy>=2.0",                  # schema 공용
    "httpx>=0.27",
    "uvicorn[standard]>=0.30",
]

[project.optional-dependencies]
prod = [
    "pymilvus>=2.4",                    # production vector store swap
    "psycopg[binary]>=3.2",             # Postgres session store swap
    "pgvector>=0.3",                    # dense index in Postgres
]
```

---

## 4. D-5 스코프: In vs Out

### D-4~D-3 POC에서 구현 (MUST)
- [ ] `schemas/agent.py`, `schemas/retrieval.py`, `schemas/eval.py`
- [ ] `retrieval/`: Chunker + Embedder + ChromaStore + BM25 + Reranker + Hybrid + Pipeline
- [ ] `agent/`: StateGraph + 4 노드 (planner·retriever·reasoner·reporter) + SQL tool
- [ ] `session/`: SessionStore (SQLite) + state_machine + events
- [ ] `eval/`: RAGAS adapter + report
- [ ] `serving/`: AnthropicClient + vLLM stub
- [ ] `api/`: FastAPI 8개 엔드포인트 + Swagger
- [ ] **샘플 데이터**: 합성 ERP 재무 데이터 10건 + 합성 판례/세법 문서 5건 (strategist `domain-storyline.md` 결과 반영)

### D-5 범위 외 (면접 후)
- [ ] `training/`: SFT + DPO 파이프라인 (30일 트랙 Week 2~4)
- [ ] `session/store.py` Postgres 구현 (SQLite로 데모 충분)
- [ ] `retrieval/vectorstore.py` Milvus 구현 (Chroma로 데모 충분)
- [ ] `api/` 인증 미들웨어 (엘리스 미니의 httpOnly 쿠키 재사용 — fork 시점)
- [ ] `differentiations/` 각 에이전트 use case

---

## 5. 면접 데모 시나리오 (5분)

1. `docker compose up` → FastAPI + ChromaDB 기동
2. `http://localhost:8000/docs` Swagger 열기
3. `POST /agents/sessions` 실행 → 세션 생성 (PENDING)
4. `POST /agents/sessions/{id}/run` → LangGraph 실행
   - 질의 예시: "특수관계자 거래 증가가 리스크 신호인가?"
   - Planner → Retriever (판례·회계기준서 hybrid search) → SQL tool (ERP 재무 조회) → Reasoner → Reporter
5. `GET /agents/sessions/{id}/events` → 전체 이벤트 리플레이 (데이터 리니지 증명)
6. `POST /eval/run` → RAGAS 점수 (faithfulness 0.8+ 목표)
7. 시연 중 "이 엔진, 엘리스 미니프로젝트 VM 상태 머신과 같은 구조. 엔티티만 VM → Agent Session으로 치환했습니다" 멘트

---

## 6. 외부 참조

- **엘리스 미니프로젝트 원본**: `C:\Users\coolb\Documents\GithubLinkedProjects\elice_cloud\README.md` — 상태 머신 5중 방어 / 이벤트 소싱 / 단가 이력 패턴 참조
- **30일 트랙**: `sandbox/tracks/30일_LLM_훈련_트랙.md` — Day 2~3 `core/retrieval/`
- **이테크시스템 JD**: `sandbox/research/job-postings/raw/2026-04-24_이테크시스템_ai-agent-rag-llm.md`
- **통합 내러티브**: `sandbox/research/job-postings/analysis/integrated-narrative.md` (strategist 작성 예정, D-2)
- **OSS 레퍼런스**: `sandbox/research/oss-references/` (researcher 작성 예정, D-2)

---

## 7. Next (D-4 첫 작업 순서)

1. Repo 루트에 `pyproject.toml` + `.python-version` (3.11)
2. `core/__init__.py` 및 7개 하위 패키지 `__init__.py` 생성
3. `schemas/` 3개 파일 구현 (Pydantic)
4. `session/` 구현 (state_machine → store → events → 단위 테스트)
5. `retrieval/` 구현 (chunker → embedder → vectorstore → bm25 → reranker → hybrid → pipeline)
6. `agent/` LangGraph 조립
7. `api/` FastAPI 라우터 + Swagger
8. `eval/` RAGAS 연결
9. 샘플 데이터 로드 + smoke 테스트

D-4 하루에 다 못 끝낼 가능성 큼. **최소 3·4·7**(schemas + session + api)이 D-4 마감 기준. Retrieval·Agent는 D-3로 밀려도 내러티브 유지됨.
