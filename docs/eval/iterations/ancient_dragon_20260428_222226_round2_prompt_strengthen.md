# Iteration · ancient_dragon · 20260428_222226

## 1. Measure

- pass_rate: **96%** (25 prompts)
- banned_violations: 0
- cjk_ratio_avg: 0.581
- cjk_han_count: 35

### History

| iteration | pass_rate |
|---|---|
| iter 5 | 90% |
| iter 6 | 90% |
| iter 7 | 96% |

## 2. Tactical analysis

### `synonym` — kg_grounding / treasure_origin

**근거**: prompt `treasure_origin` (kg_grounding): none of must_contain_any present: ['Thrór']. Reply 에서 의미 가까운 단어 후보 검토 필요.

**제안**:
```
답변: "Arkenstone 는 Vyrnaes 의 보물이었습니다. 그녀는 Erebor 의 침략 직전에 그 보석을 내려ставил."
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

- iterations 수집: **7**
- variance band (지난 3 iter): **±6.0%**
- last gain: **+6.0%**
- plateaued: **no**
- decision_ready: **no**

### 추천 axis

**`test_pool_expansion`** — variance ±6.0% — 측정 noise 가 axis 비교를 막음. prompt 수 늘려 noise band 축소.

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
    scripts/eval_iterate.py --persona ancient_dragon
```