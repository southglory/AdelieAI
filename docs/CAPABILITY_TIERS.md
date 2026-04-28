# Capability Tiers — choose what you actually need

페르소나 한 명에 필요한 기술 스택은 use case 마다 다르다. AdelieAI 는 "한 엔진" 이지만 그 안에 **5개의 능력 티어** 가 들어 있다. 각 티어는 *추가되는 기술* 과 *그것 없이 안 되는 use case* 로 정의된다.

> TL;DR: 게임 NPC 라면 **T2** 로 충분. 코드 헬퍼·고객지원이면 **T3**. 법률·의료 자문이면 **T4**. 게임 월드 시뮬레이션이면 **T5**.

## 5 스펙 축

티어는 다음 5축의 요구 강도로 정해진다:

1. **Voice 정확도** — 캐릭터에서 얼마나 안 벗어나야 하는가
2. **지식 종류** — 함축적 / 비정형 텍스트 / 구조화된 관계 / 추론 가능한 logic
3. **메모리** — 단일 turn / 최근 N turn / 요약 / 장기 기억
4. **하드웨어 타깃** — 서버 GPU / 로컬 GPU / 엔드유저 CPU / 모바일
5. **멀티 에이전트** — 단일 / 2-3 / N개 협력

## 5 티어

### T1 — Toy

| | |
|---|---|
| **대표 use case** | 챗봇 프로토타입, 간단한 대화 |
| **필요 기술** | Vanilla LLM + 시스템 프롬프트 |
| **Voice 정확도** | 시스템 프롬프트 지시 정도 (벗어남 가능) |
| **지식** | LLM 파라미터에 함축된 것만 |
| **하드웨어** | any |
| **AdelieAI** | ✅ `StubLLMClient` 또는 `TransformersClient` 단독 |

### T2 — Standard NPC ✨ *현 출하 라인*

| | |
|---|---|
| **대표 use case** | 게임 NPC, 브랜드 챗, 캐릭터 컴패니언 |
| **필요 기술** | LoRA fine-tune + 벡터 RAG (BM25 + dense + RRF + reranker) + 양자화 |
| **Voice 정확도** | 캐릭터 voice 일관 유지, 일반 질문에도 register 누설 없음 |
| **지식** | 비정형 텍스트 RAG (lore, FAQ, 문서) |
| **하드웨어** | 로컬 GPU (FP16) 또는 엔드유저 CPU (GGUF q4_k_m) |
| **AdelieAI** | ✅ `TransformersClient` + `HybridRetriever` + `GGUFClient`. v0.2 출하 산출물 |

### T3 — Vertical Advisor

| | |
|---|---|
| **대표 use case** | 코드 헬퍼, 고객지원, 도메인 어시스턴트 |
| **필요 기술** | T2 + DPO (preferred-rejected pair) + tool-use protocol (retrieval / 계산기 / 외부 API) |
| **Voice 정확도** | 정확도와 voice 의 균형 — 자문 톤 + 인용 의무 |
| **지식** | RAG 가 retrieval-as-tool 로 LLM 의 의도적 호출 대상 |
| **하드웨어** | 서버 GPU (도구 호출 latency 누적) |
| **AdelieAI** | ⚠️ DPO trainer 미구현, `core/tools/protocols.py` Protocol 만 (구현 0). v0.5 마일스톤 |

### T4 — Domain Expert

| | |
|---|---|
| **대표 use case** | 법률 자문, 의료 가이드라인, 컴플라이언스, 사내 위키 advisor |
| **필요 기술** | T3 + **RDF/OWL KG + SPARQL + OWL Reasoner** + 답변 검증 layer |
| **Voice 정확도** | hallucination 비용 매우 높음 — KG 와 axiom 기반 검증 의무 |
| **지식** | 구조화된 관계 + 추론 가능한 logic (transitive, equivalent class 등) |
| **하드웨어** | 서버 GPU + KG 서버 (Fuseki / GraphDB / Stardog) |
| **AdelieAI** | ✅ **rdflib + owlrl 통합 완료** — `core/retrieval/graph_retriever_rdflib.py` 가 Turtle 파일 (`dragon_lore.ttl`) 을 파싱해 SPARQL 1.1 (transitive `descendantOf+`) 실행. `RdflibOWLReasoner` 가 OWL-RL forward chaining 으로 subClassOf 전이 + 클래스 멤버십 추론. 의존성 없을 때는 hardcoded stub 으로 graceful degrade. |

### T5 — Multi-agent Quest

| | |
|---|---|
| **대표 use case** | 게임 월드 (5+ NPC 협력), 시뮬레이션, 멀티-페르소나 워크플로우 |
| **필요 기술** | T4 + vLLM 멀티-LoRA 동시 로드 + LangGraph 멀티-에이전트 + 페르소나별 메모리 분리 |
| **Voice 정확도** | 각 페르소나 모두 자기 voice + 다른 페르소나 와의 상호작용 일관성 |
| **지식** | 페르소나마다 자기 RAG + 공유 world state |
| **하드웨어** | 멀티 GPU (vLLM) 또는 큐 기반 단일 GPU (latency ↑) |
| **AdelieAI** | ⚠️ 단일 LangGraph 4-노드만. 멀티-LoRA · multi-agent 는 v0.6 마일스톤 (실험 09 · 11 · 12) |

## 결정 트리 — 내 use case 는 어느 티어?

```
시작
│
├─ 캐릭터 voice 가 흔들려도 "그냥 챗봇" 으로 OK?
│    └─ YES → T1
│
├─ 캐릭터 voice 일관 + 비정형 텍스트 지식 필요?
│    └─ YES, 더 필요한 거 없음 → T2 ✨ (대다수 게임 NPC, 브랜드챗)
│
├─ 도구 호출 (검색·계산·외부 API) 필요?
│    └─ YES, 정확한 관계 추론은 불필요 → T3
│
├─ "X 가 Y 의 누구냐" 같은 다중 hop 관계 추론 필요?
│  hallucination 의 법적·의료적 비용?
│    └─ YES → T4
│
└─ N명 페르소나가 한 사용자 목표에 협력?
     └─ YES → T5
```

## 페르소나 ↔ 티어 매핑 (현재 + 예정)

| persona | 산업 도메인 | target_tier | 상태 |
|---|---|---|---|
| `penguin_relaxed` | (general companion) | T2 | ✅ v0.1.5 default |
| `fish_swimmer` | (general companion) | T2 | ✅ v0.1.5 default |
| `knight_brave` | (general companion) | T2 | ✅ v0.1.5 default |
| `cynical_merchant` | 🎮 Gaming (RPG NPC) | T2 | 🔄 디자인 중 (vertical: gaming) |
| `cold_detective` | ⚖️ Legal/Investigation | T3 | 🔄 디자인 중 (vertical: legal) |
| `ancient_dragon` | 🏛️ Knowledge advisor | T4 | 🔄 디자인 중 (vertical: knowledge) |

## 서비스가 자기 티어를 어떻게 선언하는가

### 1. `/health` 의 `tier` 필드

```json
GET /health →
{
  "status": "ok",
  "llm": "Qwen/Qwen2.5-7B-Instruct+qwen-roleplay-v2",
  "tier": 2,
  "tier_max": 5,
  "tier_status": {
    "T1": "ok",
    "T2": "ok",
    "T3": "missing: tools, dpo_adapter",
    "T4": "missing: graph_retriever, owl_reasoner",
    "T5": "missing: multi_agent_runner"
  }
}
```

`tier` = 현재 빌드가 사용 가능한 최대 티어.

### 2. `Persona.target_tier` + `Persona.industry`

페르소나 자체가 자신이 어느 티어를 요구하는지 선언:

```python
Persona(
    persona_id="ancient_dragon",
    target_tier=4,
    industry="knowledge",
    ...
)
```

페르소나 갤러리 카드에 티어 배지 표시. `target_tier > current_tier` 면 *"이 페르소나는 현재 빌드에서 voice 만 작동, KG 추론은 비활성"* 표시.

### 3. `/demo/{vertical}` — 산업별 데모

| route | persona | tier 가 보여주는 것 |
|---|---|---|
| `/demo/gaming` | cynical_merchant | T2 — voice + 벡터 RAG (인벤토리 lore) |
| `/demo/legal` | cold_detective | T3 — tool-use, 인용 chip 가시화 |
| `/demo/knowledge` | ancient_dragon | T4 — KG 그래프 시각화, SPARQL trace |

각 데모 페이지는 자신의 티어를 fail-soft 함:
- T3 demo 인데 T2 까지만 빌드되어 있다면, "tool-use 비활성, voice 만 데모" 모드로 폴백.

## 티어를 켜는 방법 (extras)

`pyproject.toml` 의 optional dependencies 로 분리:

```toml
[project.optional-dependencies]
t2 = ["transformers", "peft", "trl", "rank_bm25", "chromadb", "sentence-transformers"]
t3 = ["adelie-ai[t2]"]  # + DPO 와 tools 는 코드 자체에 포함
t4 = ["adelie-ai[t2]", "rdflib", "owlready2", "pyshacl"]
t5 = ["adelie-ai[t2]", "vllm"]  # vLLM 은 Linux 한정
```

→ `pip install "adelie-ai[t4]"` 로 KG advisor 데모 활성화.
→ 미설치 시 `current_tier()` 가 자동으로 fail-soft.

## 기여자에게 — 새 페르소나 추가 시

1. 어느 티어가 필요한지 결정 (이 문서의 결정 트리)
2. `Persona(target_tier=N, industry="...")` 로 선언
3. `target_tier=N` 의 모든 의존성이 이미 AdelieAI 에 있는가?
   - YES → `personas/{persona_id}/` 디자인 후 통합
   - NO → 누락된 protocol 또는 trainer 를 먼저 구현 (별도 PR)
