# `personas/` — 산업 vertical 별 페르소나 슬롯

이 디렉터리는 **산업 도메인 별로 1명씩** 페르소나를 매핑한다. 각 페르소나는 자신의 vertical (`/demo/{slug}`) 과 자신의 capability tier 를 대표한다 (📖 [`docs/CAPABILITY_TIERS.md`](../docs/CAPABILITY_TIERS.md)).

```
personas/
├── README.md            # 이 파일
├── _template/           # 새 페르소나의 출발점 (복제 대상)
│   ├── sheet.md
│   ├── dialogue_pairs.jsonl
│   └── README.md
├── cynical_merchant/    # 🎮 gaming      — T2
├── cold_detective/      # ⚖️ legal       — T3
└── ancient_dragon/      # 🏛️ knowledge   — T4
```

## 진행 상황

| persona | vertical | tier | 상태 |
|---|---|---|---|
| `cynical_merchant` | 🎮 gaming | T2 | 🔄 디자인 대기 (Step 2) |
| `cold_detective` | ⚖️ legal | T3 | 🔄 디자인 대기 (Step 3) |
| `ancient_dragon` | 🏛️ knowledge | T4 | 🔄 디자인 대기 (Step 3) |

## 새 페르소나 추가 절차

1. `_template/` 복제 → `{persona_id}/`
2. `sheet.md` 채움: 식별자 · 정체성 · 말투 5샘플 · 시스템 프롬프트
3. `dialogue_pairs.jsonl` 에 60 역할극 + 60 일반 = 120 페어
4. `core/personas/registry.py` 에 `Persona(target_tier=N, industry="...")` 등록
5. (필요 시) `core/api/demos_router.py::VERTICALS` 에 vertical 매핑 추가
6. LoRA 학습 → `models/ours/qwen-{persona_id}-v1/`
7. judge harness 통과 → 채택

## 가이드

- [`docs/persona_design_guide.md`](../docs/persona_design_guide.md) — 60+60 페어 분포, 좋은-나쁜 페어, 함정
- [`docs/CAPABILITY_TIERS.md`](../docs/CAPABILITY_TIERS.md) — 어느 티어가 필요한지 결정 트리
- [`docs/PERSONA_PACK.md`](../docs/PERSONA_PACK.md) — `.adelie` 팩 포맷

## v0.1.5 의 기본 페르소나

`core/personas/registry.py` 에 inline 으로 정의된 3 명 (penguin / fish / knight) 은 v0.1.5 의 일반 컴패니언 페르소나. 모두 `industry="general"`, `target_tier=2`. 산업 vertical 데모와는 별도 라인 (`/web/personas` 갤러리).

## 다음 마일스톤 (실험 09 / 11 / 12)

3 vertical 이 모두 채워지면:

- **09 — vllm_multi_npc**: 3 LoRA adapter 동시 로드
- **11 — tool_use_npc**: cold_detective 의 T3 도구들 구현
- **12 — persona_orchestra**: LangGraph 으로 3 페르소나가 한 사용자 목표 협력
