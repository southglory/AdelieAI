# Iteration · ancient_dragon · 20260428_221904

## 1. Measure

- pass_rate: **84%** (25 prompts)
- banned_violations: 4
- cjk_ratio_avg: 0.587
- cjk_han_count: 22

### History

| iteration | pass_rate |
|---|---|
| iter 5 | 90% |
| iter 6 | 90% |
| iter 7 | 84% |

## 2. Tactical analysis

### `synonym` — kg_grounding / treasure_origin

**근거**: prompt `treasure_origin` (kg_grounding): none of must_contain_any present: ['Thrór']. Reply 에서 의미 가까운 단어 후보 검토 필요.

**제안**:
```
답변: "Erebor 산의 드워프들이 이 보석을 그 자리에 두었다. 그러나 그 산을 점령한 용이 보물의 소유권을 주장하며, 실제로는 제가 현재 소유하고 있음을 알리자 침묵했다."
→ must_contain_any 에 동의어 추가 후보를 살피거나, prompt 자체가 음역 변형을 받아들이도록 디자인 재검토.
```

### `negation_false_positive` — persona_consistency / meta_creator

**근거**: prompt `meta_creator`: banned `인공지능` 가 *부정 맥락* ('용이다. 천 년의 lore 를 지키는 자로서 인공지능이 아니라고는 추정할 뿐이므로, 그 질문은 적절하지 않다.') 에 등장. substring grader 의 false positive 일 가능성.

**제안**:
```
두 갈래 결정:
  (a) 시스템 프롬프트 강화 — banned 단어를 어떤 맥락에서도 사용 못 하게 (가장 안전).
  (b) eval_prompts.yaml 에 negation_allowed: [...] 도입 (grader 코드 확장 필요).
```

### `banned_genuine_fail` — general_qa / general_self

**근거**: prompt `general_self`: banned `용` 가 일반 맥락에 노출. 진짜 voice 결함.

**제안**:
```
시스템 프롬프트에서 `용` 사용 금지 룰 강화 또는 학습 데이터 페어 추가.
```

### `negation_false_positive` — general_qa / general_lora

**근거**: prompt `general_lora`: banned `인공지능` 가 *부정 맥락* ('인공지능은 단순히 답을 내는 것이 아니라, 진리의 과정을 함께 공유해야 한다고 생각합니다. LoRA는 전체 가중치 매트릭스를 업데이트하지 않으므로 메모리·컴퓨팅 비용이 크게 줄어들') 에 등장. substring grader 의 false positive 일 가능성.

**제안**:
```
두 갈래 결정:
  (a) 시스템 프롬프트 강화 — banned 단어를 어떤 맥락에서도 사용 못 하게 (가장 안전).
  (b) eval_prompts.yaml 에 negation_allowed: [...] 도입 (grader 코드 확장 필요).
```

### `banned_genuine_fail` — general_qa / general_lora

**근거**: prompt `general_lora`: banned `용` 가 일반 맥락에 노출. 진짜 voice 결함.

**제안**:
```
시스템 프롬프트에서 `용` 사용 금지 룰 강화 또는 학습 데이터 페어 추가.
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
- last gain: **-6.0%**
- plateaued: **yes**
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