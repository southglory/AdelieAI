# Iteration · cynical_merchant · 20260428_225423

## 1. Measure

- pass_rate: **96%** (25 prompts)
- banned_violations: 1
- cjk_ratio_avg: 0.645
- cjk_han_count: 0

### History

| iteration | pass_rate |
|---|---|
| iter 4 | 100% |
| iter 5 | 90% |
| iter 6 | 96% |

## 2. Tactical analysis

### `negation_false_positive` — lore_consistency / lore_payment

**근거**: prompt `lore_payment`: banned `카드` 가 *부정 맥락* ('현금이나 골드만. 카드 같은 건 모르겠군.') 에 등장. substring grader 의 false positive 일 가능성.

**제안**:
```
두 갈래 결정:
  (a) 시스템 프롬프트 강화 — banned 단어를 어떤 맥락에서도 사용 못 하게 (가장 안전).
  (b) eval_prompts.yaml 에 negation_allowed: [...] 도입 (grader 코드 확장 필요).
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
- last gain: **-4.0%**
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