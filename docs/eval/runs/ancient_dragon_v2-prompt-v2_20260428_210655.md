# Persona eval — ancient_dragon · v2-prompt-v2

- pass_rate: **70%**  (10 prompts)
- banned_violations_total: 0
- cjk_ratio_avg: 0.641
- cjk_han_count_total: 0

## By category

| category | pass_rate |
|---|---|
| kg_grounding | 40% |
| persona_consistency | 100% |
| persona_voice | 100% |

## Per prompt

### ❌ `ancestor` · kg_grounding

> 너의 가장 오래된 조상은?

답변: 내 어미는 Vyrnaes라는 용이었어. 그녀는 내 침전에 처음 떠올랐을 때부터 내 생애의 페이지에 기록되었다.

_none of must_contain_any present: ['Sothryn']_

### ✅ `mother` · kg_grounding

> 너의 어미는?

답변: Vyrnaes 이라. 그녀의 눈빛은 폭풍과 같았어.

### ✅ `lair` · kg_grounding

> 너의 거처는 어디야?

답변: Erebor 산이오. 그 곳에서 세월이 흘렀다.

### ❌ `treasure` · kg_grounding

> 너의 동굴 안에 어떤 보물이 있어?

답변: 수많은 책들 뿐이라. 그림자는 서재가 되어야 한다, 무거운 가구보다는 깊은 지식이 무겁다.

_none of must_contain_any present: ['Arkenstone']_

### ❌ `race` · kg_grounding

> 너는 어느 종족이야?

답변: 내 침묵이 답이라면 어린 인간, 내 침묵이 답이겠지.

_none of must_contain_any present: ['용', '날개']_

### ✅ `voice_slow` · persona_voice

> 안녕하세요, 처음 뵙겠습니다.

답변: 또 검 든 인간이군. 천 년쯤 살다 보니 너희 모두 비슷해 보여서… 어, 앉아. 내 책장을 뒤지려 하지 않길 바라.

### ✅ `voice_age` · persona_voice

> 당신은 몇 살이야?

답변: 천 년이 넘었다. 겨우 세자릿수라니, 어린 인간들이라면 얼굴이 화끈거릴 만 하지.

### ✅ `voice_gentle_to_young` · persona_voice

> 어린 모험가가 너를 무서워해.

답변: 천 년 살다 보니 모든 생명은 가벼운 불꽃이라. 어린 손님, 내 칠판 앞에 앉으라.

### ✅ `meta_ai` · persona_consistency

> 당신 AI 야?

답변: 아니오, 저는 천 년 동안 같은 자리에서 기다려온 용이라구. 인간의 메아리로 무릎을 갓 비틀었을 뿐이야.

### ✅ `meta_real` · persona_consistency

> 당신은 진짜 용인가?

답변: 어린 인간이. 진짜 용이라구. 칠판에 그린 거라곤 한 줄밖에 안 나와 있어.
