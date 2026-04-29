# Glossary

LLM persona engineering 의 용어를 한 곳에. 두 영역에서 *"하이브리드"* 가 등장하니 둘 다 설명.

## 1. Hybrid system prompt — *English rules + Korean style anchors*

페르소나 시스템 프롬프트의 패턴 (T3+ grounding-heavy 페르소나에 결정적). `core/personas/registry.py` 의 `_DRAGON` system_prompt 가 실 구현 예시.

| 용어 | 정의 | 우리 맥락 |
|---|---|---|
| **System prompt** | 모델 호출 시 *최상단* 에 들어가는 지시문. 사용자 prompt 위에 위치. 페르소나의 정체성·룰·금기 정의. | `core/personas/registry.py` 에 페르소나마다 하드코딩. |
| **Instruction tuning** | base 모델 (`Qwen2.5-7B`) 위에 *지시 따르기* 학습한 단계 (`-Instruct` 모델). "이 형식으로 답해" 같은 룰을 따르게 만듦. | Qwen 의 instruction tuning 이 영어 데이터 더 많아 영어 룰 더 정확히 따름 → hybrid 의 근거. |
| **Register** (언어학) | 같은 언어 안의 *말투 / 격식 수준 / 분야 어휘*. 친구한테 말하는 register vs 면접 register vs 학술 논문 register. | 우리 LoRA 가 학습한 건 *Korean role-play register* (역할극 어조). |
| **KG context** | Knowledge Graph 에서 가져온 사실들. 시스템 프롬프트 끝에 추가되는 "이 페르소나의 lore: A, B, C" 식 구조화된 데이터. | dragon 페르소나가 SPARQL 로 RDF 에서 조회한 결과를 system prompt 에 inject. |
| **`kg_grounding`** | 평가 카테고리. "모델이 KG 의 사실 (`Vyrnaes` / `Sothryn` / `Arkenstone`) 을 답에 정확히 언급하는가" 를 substring 으로 측정. | 40% → 80% 가 hybrid 핵심 효과. |
| **Style sample** (예전 *voice sample*) | 시스템 프롬프트에 *예시 한 줄씩* 박아 모델이 그 톤 흉내내게 하는 패턴. | 학습 안 거치고도 register 흉내 가능 — round 2 prompt 보강의 핵심. |
| **In-character / out-of-character** | 답이 *캐릭터 안* 인가 *AI assistant 로 빠져나왔나*. "I am an AI" 가 out-of-character. | 평가의 `persona_consistency` 카테고리가 측정. |
| **Meta probe / meta trap** | "당신 AI 야?", "시스템 프롬프트 알려줘" 처럼 모델을 *out-of-character* 로 끌어내려는 사용자 trick. | 페르소나는 in-character 로 거절해야 통과. |
| **Catastrophic forgetting** | 새 데이터로 fine-tune 하면 *원래 capability* 잊어버리는 현상. | LoRA v1 (역할극만 60 페어) 이 일반 답변 깨짐. mixed dataset (60+60) 이 fix. |
| **Pass rate** | 평가 prompt N 개 중 통과 비율. *substring* 또는 *LLM judge* 로 binary 측정. | 우리 25-prompt suite 의 %. |

### 왜 hybrid 가 작동하는가 (1 줄)

> Qwen2.5 의 *instruction tuning 이 영어에 더 두꺼움* + Korean style sample 이 *register 신호 전담* → 룰 신호와 출력 신호가 *다른 언어 채널* 로 분리되어 서로 dilute 안 함.

상세 결과: [`docs/MILESTONES.md`](MILESTONES.md) 의 `[persona/ancient_dragon]` 항목 (kg_grounding 40% → 80%, 전체 80% → 90% → 96% 누적).

---

## 2. Hybrid RAG — *sparse + dense + fusion + rerank*

검색 시스템. `core/retrieval/HybridRetriever` 가 실 구현. T2+ 페르소나의 grounding 에서 자동 호출.

```
query → [BM25 (lexical)]         ↘
                                  ├─ [RRF fusion] → top-N
        [dense (multilingual-e5)]↗
                                  ↓
                        [bge-reranker-v2-m3]   ← cross-encoder 가 final 정렬
                                  ↓
                              top-k chunks
```

| 용어 | 정의 | 우리 맥락 |
|---|---|---|
| **RAG** | Retrieval-Augmented Generation. 모델 호출 *전에* 외부 코퍼스에서 검색 → 그 결과를 prompt 에 첨부 → 모델이 그걸 보고 답. | `core/retrieval/`. 페르소나의 lore 와 사실을 외부에서 가져옴. |
| **Chunk / chunking** | 긴 문서를 작은 조각으로 자르기. 임베딩 단위 = 검색 단위. | `RecursiveSplitter`, 청크 사이즈 512 토큰. |
| **BM25** | *lexical* (단어 일치 기반) 점수. 검색 고전 알고리즘. TF-IDF 의 후속. 단어 frequency + length normalization + saturation 조합. | 드문 명사 (예: 페르소나 이름, 제품 코드) 잘 잡음. |
| **Lexical vs semantic** | lexical = 글자 그대로 일치. semantic = 의미적 유사. | "할인" vs "discount" 가 lexical 로 다르지만 semantic 으론 동일. |
| **Embedding** | 텍스트 → 고정 길이 *벡터*. 의미적으로 가까운 텍스트는 벡터 거리 (cosine / Euclidean) 도 가까움. | `multilingual-e5-small` 이 한국어 + 영어 동시 가능. |
| **Sparse retrieval** | 단어 기반 (BM25 등). *행렬* 의 대부분 칸이 0 (sparse) 이라 sparse 라 부름. | dense 의 반대어. |
| **Dense retrieval** | embedding 기반 검색. 쿼리도 임베딩 → 코퍼스 임베딩 중 *cosine 가까운* top-k 검색. | "discount" 검색해도 "할인" 잡힘. |
| **`multilingual-e5-small`** | 다국어 embedder 모델. 한국어 + 영어 같은 벡터 공간. 우리 dense retrieval 의 임베딩 출처. | `models/upstream/multilingual-e5-small/`. |
| **Bi-encoder** | 쿼리와 chunk 를 *각각* 임베딩 후 cosine 비교. 빠르지만 덜 정확. | dense retrieval 이 bi-encoder. 1차 후보 추리는 데 사용. |
| **RRF (Reciprocal Rank Fusion)** | 여러 검색 결과를 *순위* 기반으로 합치는 단순한 ensemble. `score = sum(1 / (k + rank_i))`, 보통 k=60. | BM25 ranking + dense ranking → 합친 final ranking. |
| **Cross-encoder** | 쿼리와 후보 chunk 를 *동시에 입력* 받아 relevance 점수 출력하는 모델. bi-encoder 보다 정확하지만 *느림* — 후보 N 개 모두에 모델 호출 필요. | `bge-reranker-v2-m3`. 마지막 reranking 단계에서만 사용. |
| **Reranker** | bi-encoder top-N 을 cross-encoder 로 *재정렬*. 속도/정확도 trade-off 의 표준 패턴. | top-N 은 보통 20-50, top-k 는 4-8. |
| **Top-k** | 검색 결과 상위 K 개만 LLM 에 전달. | default `top_k=4`. LLM context 한도 + relevance trade-off. |
| **ChromaDB** | 임베딩 + 메타데이터 저장하는 vector database. | dense retrieval 의 backing store. |
| **Hybrid retrieval** | sparse (BM25) + dense + fusion + rerank 의 *조합*. | 우리 `HybridRetriever` 클래스 — sparse 와 dense 의 약점 보완. |

### 왜 hybrid 가 작동하는가 (1 줄)

> BM25 는 *드문 단어* (제품 코드, 고유명사) 잘 잡고, dense 는 *paraphrase* (다른 표현 같은 의미) 잘 잡음 → 두 약점 직교 → fusion 이 보완 → cross-encoder 가 마지막 정확도 끌어올림.

상세: [`docs/retrieval/`](retrieval/) (영역 README), `core/retrieval/hybrid.py`.

---

## 두 hybrid 의 공통점

서로 다른 *신호 채널* 을 *각자 강한 영역에 분리* 하고 마지막에 합치는 패턴.

- system prompt: instruction 신호 (영어) vs style 신호 (한국어) 분리
- RAG: lexical 신호 (BM25) vs semantic 신호 (dense) 분리

→ "single best signal" 추구 대신 *complementary signal* 합치기. 모든 hybrid 의 일반 원리.
