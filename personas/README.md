# `personas/` — 사용자가 채울 페르소나 슬롯

이 디렉터리는 **빈 슬롯**과 **템플릿**을 보관한다. 실제 페르소나 디자인은 사용자 영역.

```
personas/
├── README.md            # 이 파일
├── _template/           # 새 페르소나의 출발점 (복제 대상)
│   ├── sheet.md         # 캐릭터 시트 (식별자/성격/말투/시스템 프롬프트)
│   ├── dialogue_pairs.jsonl  # 학습 페어 (예시 5줄)
│   └── README.md        # 사용법
├── npc1/                # 미설계 슬롯 1
├── npc2/                # 미설계 슬롯 2
├── npc3/                # 미설계 슬롯 3
├── npc4/                # 미설계 슬롯 4
└── npc5/                # 미설계 슬롯 5
```

## 진행 상황

| 슬롯 | persona_id | display_name | 상태 |
|---|---|---|---|
| npc1 | _ | _ | 디자인 대기 |
| npc2 | _ | _ | 디자인 대기 |
| npc3 | _ | _ | 디자인 대기 |
| npc4 | _ | _ | 디자인 대기 |
| npc5 | _ | _ | 디자인 대기 |

각 슬롯이 채워지면 디렉터리명을 `npc{N}` 에서 실제 `persona_id` 로 rename 한다 (예: `npc1` → `rogue_jester`).

## 가이드

- [`docs/persona_design_guide.md`](../docs/persona_design_guide.md) — 60+60 페어 분포 / 좋은-나쁜 페어 / 함정 7개
- [`docs/PERSONA_PACK.md`](../docs/PERSONA_PACK.md) — `.adelie` 팩 포맷 (디자인이 학습되어 .adelie 가 되는 단계)
- [`docs/TRAINING.md`](../docs/TRAINING.md) — hyperparameter / v1→v2 학습 사이클

## v0.1.5 의 기본 페르소나는 어디?

`core/personas/registry.py` 에 inline 으로 정의된 3 명 (penguin / fish / knight) 은 v0.1.5 demo 용. v0.2 에서도 그대로 작동. 새로 디자인하는 5명은 v0.3 의 `.adelie` 팩 자동 발견 (auto-discovery) 으로 합류 예정.

## 다음 마일스톤 (실험 09 / 11 / 12)

5명을 모두 채우면 다음 실험들의 입력이 된다 (사이드 differentia-llm 의 `experiments/INDEX.md` 참고):

- **09 — vllm_multi_npc**: 5 LoRA adapter 를 vLLM 한 GPU 에 동시 로드
- **11 — tool_use_npc**: 각 페르소나에 도구(웹검색·계산기·DB) 부여
- **12 — persona_orchestra**: LangGraph 으로 5명이 한 사용자 목표에 협력
