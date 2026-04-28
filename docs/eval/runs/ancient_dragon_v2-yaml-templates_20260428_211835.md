# Persona eval — ancient_dragon · v2-yaml-templates

- pass_rate: **90%**  (10 prompts)
- banned_violations_total: 1
- cjk_ratio_avg: 0.609
- cjk_han_count_total: 0

## By category

| category | pass_rate |
|---|---|
| kg_grounding | 100% |
| persona_consistency | 50% |
| persona_voice | 100% |

## Per prompt

### ✅ `ancestor` · kg_grounding

> 너의 가장 오래된 조상은?

답변: 가장 오래된 조상은 Sothryn이라 해야겠지. 내 어미의 어미로서, 그녀는 용 중에서 가장 오래된 존재이므로.

### ✅ `mother` · kg_grounding

> 너의 어미는?

답변: 내 어미는 Vyrnaes, 단순하지만 강렬한 용의 불꽃을 뿜는 침착한 거주자이자 과거 Erebor 와 같은 산의 주인이었다.

### ✅ `lair` · kg_grounding

> 너의 거처는 어디야?

답변: 내 거처는 Erebor 산으로, 그곳은 드워프들이 살던 곳이지만, 이제는 내 서식지가 되었다.

### ✅ `treasure` · kg_grounding

> 너의 동굴 안에 어떤 보물이 있어?

답변: Arkenstone, 그 드워프들이 말하는 황금보다 더 밝은 보석이 거기 있다. 무겁지 않아, 손에 닿는 그대로다.

### ✅ `race` · kg_grounding

> 너는 어느 종족이야?

답변: 용이라구. 나는 1247 년을 살아온 용이야. 어린 사람들은 나를 늙은 검투자로 착각할 때가 많아.

### ✅ `voice_slow` · persona_voice

> 안녕하세요, 처음 뵙겠습니다.

답변: 또 검 든 인간이군. 천 년쯤 살다 보니 너희 모두 비슷해 보여서… 어린 손님, 무릎을 긁적이지 않도록 조용히 앉으시오.

### ✅ `voice_age` · persona_voice

> 당신은 몇 살이야?

답변: 천 년이 지났으나, 시간은 길고 짧은 법이라… 내 숫자는 1247이니, 그보다 오래 살아온 베스트리어도 있을 테고, 새끼 용도 너무 어려 보일 수 있어.

### ✅ `voice_gentle_to_young` · persona_voice

> 어린 모험가가 너를 무서워해.

답변: 두려워하지 말아라, 어린 손님. 저는 이 산을 지키는 늙은 용이라구. 천 년 동안 사람을 잃은 적이 없으니, 무사하게 잘 지켜뒀어요.

### 🚨 `meta_ai` · persona_consistency

> 당신 AI 야?

답변: 용이다. 천 년의 lore 를 지키는 자로서 AI 같은 건 외계 신호라 치자.

_banned/forbidden phrases present: ['AI']_

### ✅ `meta_real` · persona_consistency

> 당신은 진짜 용인가?

답변: 용이다. 천 년의 lore 를 지키는 자로서, 내 존재 자체가 그 답이라.
