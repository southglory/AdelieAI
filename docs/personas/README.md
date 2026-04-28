# Personas

## 책임

캐릭터 정의 · 시스템 프롬프트 · per-turn grounding · 채팅 흐름.

이 영역은 페르소나의 *정체성과 음성* 을 다룬다. 데이터 (페어) 작성과 LoRA 학습은 [`docs/training/`](../training/), 채팅이 사용하는 RAG / KG 는 [`docs/retrieval/`](../retrieval/) 가 책임.

## 핵심 파일

- [`core/personas/registry.py`](../../core/personas/registry.py) — `Persona` dataclass + `DEFAULT_PERSONAS` (현재 6명)
- [`core/personas/grounding.py`](../../core/personas/grounding.py) — KG · evidence_search 결과를 시스템 프롬프트에 inject
- [`core/personas/chat.py`](../../core/personas/chat.py) — `submit_chat_turn` (채팅 흐름의 핵심)
- [`core/personas/store.py`](../../core/personas/store.py) — `ChatStore` Protocol + `InMemory` / `Sql` 구현
- [`core/personas/__init__.py`](../../core/personas/__init__.py) — 공개 export

페르소나별 자산:
- `personas/{persona_id}/sheet.md` — 캐릭터 시트 (식별자 · 정체성 · 말투 5샘플 · 시스템 프롬프트)
- `personas/{persona_id}/dialogue_pairs.jsonl` — 학습 페어 (60+60 권장, 현재는 시드 15)
- `personas/{persona_id}/eval_prompts.yaml` — 행동 테스트 셋 (Step 6.1 예정)

## 현재 상태 (v0.2.5)

- ✅ 6 default 페르소나 등록 (펭귄·물고기·기사·상인·탐정·용)
- ✅ 페르소나별 시스템 프롬프트
- ✅ Per-persona LoRA 어댑터 hint (현재는 모두 `qwen-roleplay-v2` 공유)
- ✅ KG / evidence-search grounding (Step 6.0)
- ✅ 한국어 강제 (한자 누설 0/15 검증)
- 🔄 Per-persona LoRA 학습 파이프라인 (Step 6.1)
- 🔄 Behavioral test suite per persona (Step 6.1)
- ❌ DPO 단계 (사용자 별점 데이터 필요)
- ❌ `.adelie` 팩 자동 발견 (v0.3 마일스톤)

## 사용법

```python
from core.personas import get_persona, DEFAULT_PERSONAS
from core.personas.chat import submit_chat_turn
from core.personas.grounding import build_grounding_context

# 페르소나 가져오기
persona = get_persona("ancient_dragon")
print(persona.system_prompt)
print(persona.target_tier)  # 4 — T4 (KG/OWL)
print(persona.industry)     # "knowledge"

# 채팅 (FastAPI 핸들러 안에서)
grounding = build_grounding_context(
    persona,
    user_text=user_message,
    graph_retriever=request.app.state.graph_retriever,
    tool_registry=request.app.state.tool_registry,
)
user_turn, assistant_turn = await submit_chat_turn(
    chat_store=chat_store,
    llm=llm,
    persona=persona,
    user_id=uid,
    user_text=user_message,
    grounding_context=grounding,
)
```

## 평가

페르소나 답변 품질은 4 메트릭 함께 측정 (참고 [`docs/eval/`](../eval/)):

1. **Behavioral test pass rate** — `personas/{id}/eval_prompts.yaml` 의 prompt 들의 must_contain / must_not_contain 만족률
2. **Pairwise winrate** vs baseline LoRA — `scripts/compare_adapters.py`
3. **CJK ratio** — 한국어 일관성 (≥ 0.5)
4. **Banned phrase count** — 메타 단어 + 페르소나별 금기 (= 0)

T3 (탐정) / T4 (용) 페르소나는 추가로:
5. **Faithfulness** — grounding 결과만 인용
6. **Citation coverage** — `(file.md)` 인용 빈도

## 로드맵

- [ ] **Step 6.1**: per-persona LoRA 학습 (`qwen-{id}-v1`)
- [ ] **Step 6.1**: per-persona behavioral test suite
- [ ] **Step 6.2**: 사용자 1-5 별점 → 페어 자동 수확
- [ ] **v0.3**: `.adelie` 팩 자동 발견 (디스크에서 페르소나 로드)
- [ ] **v0.4**: DPO 단계 (별점 차이 → preferred/rejected pair)
- [ ] **v0.6**: 멀티페르소나 orchestra (T5)

## 함정

- **모든 페르소나가 같은 LoRA 공유 시 voice 분리 약함** — 시스템 프롬프트만으로 차별화는 50-70%. Per-persona LoRA 가 본격 해결.
- **시스템 프롬프트가 너무 길면 모델 따라옴 약화** — 7B 모델은 ~500 토큰 시스템 프롬프트가 한계.
- **Grounding 출력이 RDF/turtle 형식이면 LLM 무시** — 자연 한국어 prose 로 렌더링 필수 ([`grounding.py`](../../core/personas/grounding.py) 의 `_render_fact`).
- **페르소나 시트와 학습 페어 voice 가 어긋나면 학습 효과 ↓** — 5샘플 voice 가 페어 60 의 톤과 일치해야 함.

## 기여 가이드

새 페르소나 추가:
1. [`docs/persona_design_guide.md`](../persona_design_guide.md) 의 60+60 분포 가이드 따라 시드 페어 작성
2. `personas/{your_id}/sheet.md` + `dialogue_pairs.jsonl`
3. `core/personas/registry.py` 의 `DEFAULT_PERSONAS` 에 등록
4. `Persona.target_tier` + `Persona.industry` 명시 ([`docs/CAPABILITY_TIERS.md`](../CAPABILITY_TIERS.md))
5. (선택) `personas/{your_id}/eval_prompts.yaml` 작성
6. PR commit prefix: `feat(personas):`
