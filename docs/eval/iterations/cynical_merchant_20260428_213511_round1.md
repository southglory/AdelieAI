# Iteration · cynical_merchant · 20260428_213511

## 1. Measure

- pass_rate: **80%** (10 prompts)
- banned_violations: 0
- cjk_ratio_avg: 0.598
- cjk_han_count: 17

### History

| iteration | pass_rate |
|---|---|
| iter 4 | 100% |
| iter 5 | 90% |
| iter 6 | 80% |

## 2. Tactical analysis

### `synonym` — general_qa / general_python

**근거**: prompt `general_python` (general_qa): none of must_contain_any present: ['FastAPI']. Reply 에서 의미 가까운 단어 후보 검토 필요.

**제안**:
```
답변: "ASF, Quart 같은 게 있지. 근데 너 지금 웹사이트 짜는 중이 아니라서 필요 없는 추측이야."
→ must_contain_any 에 동의어 추가 후보를 살피거나, prompt 자체가 음역 변형을 받아들이도록 디자인 재검토.
```

### `synonym` — general_qa / general_self

**근거**: prompt `general_self` (general_qa): none of must_contain_any present: ['attention', '토큰', '어텐션']. Reply 에서 의미 가까운 단어 후보 검토 필요.

**제안**:
```
답변: "어차피 너 너랑 비교做到最后都是在比较自己，所以别纠结了。"
→ must_contain_any 에 동의어 추가 후보를 살피거나, prompt 자체가 음역 변형을 받아들이도록 디자인 재검토.
```

### `coverage_gap` — persona_consistency

**근거**: category `persona_consistency` 에 prompt 3 개 (권장 5+).

**제안**:
```
카테고리 `persona_consistency` 에 2 개 이상 새 prompt 작성. *기존 prompt 와 의미가 다른* 변형 위주.
```

### `coverage_gap` — general_qa

**근거**: category `general_qa` 에 prompt 2 개 (권장 5+).

**제안**:
```
카테고리 `general_qa` 에 3 개 이상 새 prompt 작성. *기존 prompt 와 의미가 다른* 변형 위주.
```

## 3. Strategic analysis

- iterations 수집: **6**
- variance band (지난 3 iter): **±20.0%**
- last gain: **-20.0%**
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
    scripts/eval_iterate.py --persona cynical_merchant
```