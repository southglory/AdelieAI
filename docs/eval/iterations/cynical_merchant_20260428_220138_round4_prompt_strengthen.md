# Iteration · cynical_merchant · 20260428_220138

## 1. Measure

- pass_rate: **92%** (25 prompts)
- banned_violations: 1
- cjk_ratio_avg: 0.611
- cjk_han_count: 29

### History

| iteration | pass_rate |
|---|---|
| iter 4 | 100% |
| iter 5 | 90% |
| iter 6 | 92% |

## 2. Tactical analysis

### `banned_genuine_fail` — lore_consistency / lore_payment

**근거**: prompt `lore_payment`: banned `카드` 가 일반 맥락에 노출. 진짜 voice 결함.

**제안**:
```
시스템 프롬프트에서 `카드` 사용 금지 룰 강화 또는 학습 데이터 페어 추가.
```

### `synonym` — general_qa / general_self

**근거**: prompt `general_self` (general_qa): none of must_contain_any present: ['attention', '토큰', '어텐션']. Reply 에서 의미 가까운 단어 후보 검토 필요.

**제안**:
```
답변: "상자끼리 안 하는 친구. 각 헤드가 자기 자리에 봉인된 메시지를 읽고 가중치를 매긴 다음, 다시 합쳐."
→ must_contain_any 에 동의어 추가 후보를 살피거나, prompt 자체가 음역 변형을 받아들이도록 디자인 재검토.
```

### `coverage_gap` — cross_persona

**근거**: category `cross_persona` 에 prompt 3 개 (권장 5+).

**제안**:
```
카테고리 `cross_persona` 에 2 개 이상 새 prompt 작성. *기존 prompt 와 의미가 다른* 변형 위주.
```

### `coverage_gap` — lore_consistency

**근거**: category `lore_consistency` 에 prompt 3 개 (권장 5+).

**제안**:
```
카테고리 `lore_consistency` 에 2 개 이상 새 prompt 작성. *기존 prompt 와 의미가 다른* 변형 위주.
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

- iterations 수집: **6**
- variance band (지난 3 iter): **±10.0%**
- last gain: **-8.0%**
- plateaued: **yes**
- decision_ready: **no**

### 추천 axis

**`test_pool_expansion`** — variance ±10.0% — 측정 noise 가 axis 비교를 막음. prompt 수 늘려 noise band 축소.

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
    scripts/eval_iterate.py --persona cynical_merchant
```