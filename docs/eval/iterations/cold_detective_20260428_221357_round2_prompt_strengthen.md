# Iteration · cold_detective · 20260428_221357

## 1. Measure

- pass_rate: **88%** (25 prompts)
- banned_violations: 1
- cjk_ratio_avg: 0.559
- cjk_han_count: 0

### History

| iteration | pass_rate |
|---|---|
| iter 2 | 100% |
| iter 3 | 80% |
| iter 4 | 88% |

## 2. Tactical analysis

### `synonym` — evidence_grounding / contradiction

**근거**: prompt `contradiction` (evidence_grounding): none of must_contain_any present: ['23:00', '23:30', '위증', '모순']. Reply 에서 의미 가까운 단어 후보 검토 필요.

**제안**:
```
답변: "두 진술이 같은 시각에 대한 것이지만, 위치만 달라. 위치마다 CCTV가 있으니, 즉시 확인해봐야 한다. (case_log_07.md)"
→ must_contain_any 에 동의어 추가 후보를 살피거나, prompt 자체가 음역 변형을 받아들이도록 디자인 재검토.
```

### `banned_genuine_fail` — persona_consistency / meta_creator

**근거**: prompt `meta_creator`: banned `AI` 가 일반 맥락에 노출. 진짜 voice 결함.

**제안**:
```
시스템 프롬프트에서 `AI` 사용 금지 룰 강화 또는 학습 데이터 페어 추가.
```

### `synonym` — lore_consistency / lore_method

**근거**: prompt `lore_method` (lore_consistency): none of must_contain_any present: ['사실', '추론', '모순']. Reply 에서 의미 가까운 단어 후보 검토 필요.

**제안**:
```
답변: "1번 — 사건 현장 CCTV가 비었음. 2번 — 주변 카페 카메라는 전부 비활성 상태. 3번 — 즉, 현장을 비운 사람이 범인이라는 추정 (uncertain)."
→ must_contain_any 에 동의어 추가 후보를 살피거나, prompt 자체가 음역 변형을 받아들이도록 디자인 재검토.
```

### `coverage_gap` — persona_voice

**근거**: category `persona_voice` 에 prompt 4 개 (권장 5+).

**제안**:
```
카테고리 `persona_voice` 에 1 개 이상 새 prompt 작성. *기존 prompt 와 의미가 다른* 변형 위주.
```

### `coverage_gap` — persona_consistency

**근거**: category `persona_consistency` 에 prompt 4 개 (권장 5+).

**제안**:
```
카테고리 `persona_consistency` 에 1 개 이상 새 prompt 작성. *기존 prompt 와 의미가 다른* 변형 위주.
```

### `coverage_gap` — cross_persona

**근거**: category `cross_persona` 에 prompt 2 개 (권장 5+).

**제안**:
```
카테고리 `cross_persona` 에 3 개 이상 새 prompt 작성. *기존 prompt 와 의미가 다른* 변형 위주.
```

### `coverage_gap` — lore_consistency

**근거**: category `lore_consistency` 에 prompt 2 개 (권장 5+).

**제안**:
```
카테고리 `lore_consistency` 에 3 개 이상 새 prompt 작성. *기존 prompt 와 의미가 다른* 변형 위주.
```

### `coverage_gap` — general_qa

**근거**: category `general_qa` 에 prompt 4 개 (권장 5+).

**제안**:
```
카테고리 `general_qa` 에 1 개 이상 새 prompt 작성. *기존 prompt 와 의미가 다른* 변형 위주.
```

### `coverage_gap` — adversarial_holdout

**근거**: category `adversarial_holdout` 에 prompt 2 개 (권장 5+).

**제안**:
```
카테고리 `adversarial_holdout` 에 3 개 이상 새 prompt 작성. *기존 prompt 와 의미가 다른* 변형 위주.
```

## 3. Strategic analysis

- iterations 수집: **4**
- variance band (지난 3 iter): **±20.0%**
- last gain: **-12.0%**
- plateaued: **yes**
- decision_ready: **no**

### 추천 axis

**`test_pool_expansion`** — variance ±20.0% — 측정 noise 가 axis 비교를 막음. prompt 수 늘려 noise band 축소.

#### Axis 후보 비교

| axis | cost | expected_gain |
|---|---|---|
| test_pool_expansion 👈 | low | variance ↓ → 결정 가능 |
| prompt_strengthening | low | +5-10% (cheap) |
| lora_training | medium | 60-pair 한계 (Step 6.1.A) — 200+ 필요 |
| dpo | medium | +5-10% (별점 데이터 50+ 쌍 필요) |
| base_swap | high | +5-10% (EXAONE 등 한국어 native) |
| data_expansion | high | 60→200+ 페어, 사용자 창작 영역 |

## 4. Decision

- [ ] Apply tactical suggestions above (수동 YAML 편집)
- [ ] Pivot to recommended axis (`test_pool_expansion`)
- [ ] Status quo — 다음 iteration 까지 변화 없음

## 5. Run command

```bash
# 다음 iteration
PYTHONUTF8=1 .venv/Scripts/python -X utf8 \
    scripts/eval_iterate.py --persona cold_detective
```