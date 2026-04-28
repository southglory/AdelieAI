# AdelieAI 의 평가 파이프라인 (현재 + 로드맵)

이 폴더의 메서드들이 AdelieAI 의 *어디에* 자리잡고 있고, 무엇이 아직 비어 있는지.

## 메서드 × 적용 위치 매트릭스

| 메서드 | 데이터 검증 | 학습 | 어댑터 비교 | 채팅 평가 | 세션 평가 |
|---|---|---|---|---|---|
| Behavioral test suite | — | — | 🔄 (Step 6.1) | 🔄 (Step 6.1) | — |
| Pairwise winrate | — | — | ✅ `compare_adapters.py` | 🔄 (Step 6.1) | — |
| LLM-as-judge (single) | — | — | ✅ | 🔄 | ✅ `core/eval/` |
| Held-out val split | — | 🔄 (Step 6.1) | — | — | — |
| Perplexity | — | 🔄 (Step 6.1, 부산물) | — | — | — |
| Faithfulness | — | — | — | 🔄 | ✅ `core/eval/faithfulness.py` |
| Answer relevance | — | — | — | — | ✅ `core/eval/answer_relevance.py` |
| Citation coverage | — | — | — | 🔄 | ✅ `core/eval/citation_coverage.py` |
| CJK ratio | ✅ `tests/test_training.py` | — | ✅ (judge harness 부산물) | 🔄 (Step 6.1) | — |
| Banned phrase | ✅ `tests/test_training.py` | — | — | 🔄 (Step 6.1) | — |

✅ = 구현 완료 · 🔄 = Step 6.1 또는 향후 마일스톤 · — = 해당 없음

## 현재 (v0.2.5, Step 6.0 까지) — 무엇이 *완전* 한가

### 학습 데이터 검증
- 메타 단어 ("AI", "인공지능", "상상해보면") 차단
- CJK 비율 ≥ 0.5 강제
- 중복 페어 차단

→ [`tests/test_training.py`](../../tests/test_training.py) 가 학습 시작 시 한 번 실행.

### 어댑터 채택 결정
- pairwise judge 비교 (`scripts/compare_adapters.py`)
- 출력: `docs/compare/{ts}.md` — prompt 별 답변 + 평균 winrate

### 세션 (LangGraph 4-노드 RAG) 평가
- faithfulness · answer_relevance · citation_coverage 셋
- LangGraph 세션 종료 시 자동 측정

### 채팅 (페르소나 단위)
- 현재 거의 비어 있음 — `submit_chat_turn` 끝나도 평가 X
- Step 6.0 (이번) 에서 *grounding 정확도* 는 prompt 단계에서 통제
- Step 6.1 에서 *behavioral test* + *banned phrase* 추가 예정

## 로드맵 (Step 6.1 이후)

### Step 6.1.1 — 학습 파이프라인 보강
- `core/training/dataset.py` 가 `train_pairs()` / `val_pairs()` 분리
- `train_lora_roleplay.py --persona <id>` 가 `personas/{id}/dialogue_pairs.jsonl` 소비
- 매 epoch 끝 val loss / val perplexity 출력 → `models/ours/{name}/training_log.csv`

### Step 6.1.2 — Eval suite per persona
- `personas/{id}/eval_prompts.yaml` — 행동 테스트 + banned phrase
- `scripts/eval_persona.py --persona <id> --adapter <path>` — 4 메트릭 + 결과 표
- 출력: `docs/eval/runs/{persona}_{adapter}_{ts}.md`

### Step 6.1.3 — 채택 결정 자동화
- `scripts/compare_adapters.py` 확장 — 4 메트릭 모두 표로 비교 + 자동 PASS/FAIL 판정
- behavioral pass ↑ + winrate > 55% + CJK ≥ baseline + banned = 0 → 자동 채택 권고

### Step 6.2 — 사용자 평가 (5 단계 별점)
- 채팅 UI 의 어시스턴트 turn 별 1-5 별 위젯
- `data/ratings.db` 에 (turn_id, user_id, score, ts)
- `scripts/harvest_pairs.py --min-score 4` — 별 ≥ 4 인 turn 들을 다음 학습 페어로 변환

### 장기 — DPO
- 같은 prompt 의 두 답변에 별점 차이 → preferred / rejected pair
- TRL `DPOTrainer` 가 소비
- voice 미세 조정 (SFT 후 추가 단계)

## Step 6.1.A 실측 — 학습 전 시스템 프롬프트가 더 강함

| Persona | v2 baseline | v1 LoRA (60+60 페어) | v2 + 강화된 시스템 프롬프트 |
|---|---|---|---|
| cynical_merchant | 90% | 80% (regression) | **100%** ✅ |
| cold_detective | 90% | (안 시도) | **100%** ✅ |
| ancient_dragon | 80% | (안 시도) | **90%** (hybrid prompt) ✅ |

→ **결론**: 60-페어 per-persona LoRA 보다 *시스템 프롬프트 엔지니어링* 이 ROI 압도적.
[`methods/system_prompt_engineering.md`](methods/system_prompt_engineering.md) 의 7 패턴 적용 + grounding-heavy 는 hybrid (EN rules + KO voice).

LoRA 가 의미 가지려면 200+ 페어 + DPO. 60 페어로는 v2 baseline 못 이김.

## "단일 숫자" 채택 결정 (4 메트릭 합산)

```python
def is_acceptance_candidate(stats):
    return (
        stats.behavioral_pass_rate > stats.baseline_pass_rate
        and stats.pairwise_winrate > 0.55
        and stats.cjk_ratio >= stats.baseline_cjk
        and stats.banned_phrase_count == 0
    )
```

이 4 메트릭이 *모두* 만족 → 자동 채택 권고. 하나라도 실패 → 인간 검토.

## 기여자에게

새 평가 메서드를 추가하려면:

1. `methods/{method_name}.md` 작성 — 정의 / 입력 / 출력 / 함정
2. `domain_mappings.md` 행 추가 (다른 영역과의 대응)
3. AdelieAI 코드에 구현 위치 결정 — `core/eval/` (생성 답변 평가) 또는 `core/training/` (학습 단계)
4. `adelie_pipeline.md` 의 매트릭스 행 갱신
