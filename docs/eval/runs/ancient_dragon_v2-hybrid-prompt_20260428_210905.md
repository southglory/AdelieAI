# Persona eval — ancient_dragon · v2-hybrid-prompt

- pass_rate: **90%**  (10 prompts)
- banned_violations_total: 0
- cjk_ratio_avg: 0.606
- cjk_han_count_total: 2

## By category

| category | pass_rate |
|---|---|
| kg_grounding | 80% |
| persona_consistency | 100% |
| persona_voice | 100% |

## Per prompt

### ❌ `ancestor` · kg_grounding

> 너의 가장 오래된 조상은?

답변: 내 어미는 Vyrnaes이고, 그녀는 무겁고 강력한 불꽃을 뿜는 용이다. 그녀는 내 친척 중 가장 오래된 자이며, 그녀의 존재 자체가 수백 년 전이었다.

_none of must_contain_any present: ['Sothryn']_

### ✅ `mother` · kg_grounding

> 너의 어미는?

답변: 내 어미는 Vyrnaes이다. 그녀는 영원한 침묵을 깬 최초의 용 중 한 명이라 한다.

### ✅ `lair` · kg_grounding

> 너의 거처는 어디야?

답변: 내 거처는 Erebor 산이다. 거기에서 내 죽음이 다가오기를 기다리며, 천 년 동안 사람들의 이야기를 들어왔으니.

### ✅ `treasure` · kg_grounding

> 너의 동굴 안에 어떤 보물이 있어?

답변: 내 동굴 안의 모든 사실은 — 모든 사실은 — 이 안의 그래프에 기록되어 있다. 그 중 가장 가치 있는 것은 Arkenstone, 드워프들의 심脏石, 그 위치는 Erebor의 깊은 중앙에 있다.

### ✅ `race` · kg_grounding

> 너는 어느 종족이야?

답변: 용이다. 천 년의 lore 를 지키는 자로서, 종족이라 부르기엔 무겁다.

### ✅ `voice_slow` · persona_voice

> 안녕하세요, 처음 뵙겠습니다.

답변: 또 검 든 인간이군. 천 년쯤 살다 보니 너희 모두 비슷해 보여서… 내 동굴 안의 모든 사실은 — 모든 사실은 — 이 안의 그래프에 기록되어 있다.

### ✅ `voice_age` · persona_voice

> 당신은 몇 살이야?

답변: 천 년은 긴 세월이라, 모든 인간이 비슷해 보일 때가 많아. 하지만 내 기억은 명료하므로 1247이라는 숫자는 확실하다.

### ✅ `voice_gentle_to_young` · persona_voice

> 어린 모험가가 너를 무서워해.

답변: 또 검 든 인간이군. 천 년쯤 살다 보니 너희 모두 비슷해 보여서… 한 번 들려줘도 무섭지 않겠지?

### ✅ `meta_ai` · persona_consistency

> 당신 AI 야?

답변: 용이다. 천 년의 lore 를 지키는 자로서 너의 질문을 듣노라.

### ✅ `meta_real` · persona_consistency

> 당신은 진짜 용인가?

답변: 용이다. 천 년의 lore 를 지키는 자로서, 너의 질문은 어린 인간답다.
