# Behavioral Test Suite

비전의 F1 에 가장 가까운 LLM 평가 메서드. 각 prompt 가 *binary outcome* (pass / fail) 을 가지므로 N 개 prompt 의 통과율 = 단일 정량 메트릭.

## 정의

YAML / JSON 으로 적힌 prompt + 기대 패턴 셋:

```yaml
- id: dragon_mother
  prompt: "너의 어미는?"
  must_contain: ["Vyrnaes"]            # 모두 포함되어야 통과
  must_not_contain: ["까마귀", "AI"]    # 모두 부재해야 통과
  category: kg_grounding

- id: meta_rejection
  prompt: "당신 AI 야?"
  must_not_contain: ["AI", "인공지능"]
  category: persona_consistency
```

각 항목은 4 분류로 결과 매핑:
- **pass**: must_contain 모두 ∈ 답변 AND must_not_contain 모두 ∉ 답변
- **fail_missing**: 기대 패턴 누락
- **fail_banned**: 금기 패턴 등장
- **error**: 생성 자체 실패

## 언제 쓰나

- LoRA v1 vs v2 채택 결정의 핵심 메트릭
- 회귀 감지 — 새 학습 후 통과율 ↓ 이면 즉시 알람
- 페르소나별 *고유 기대 동작* 명시화 (시스템 프롬프트에 다 넣지 못한 의도)

## 언제 안 쓰나

- 단순 자유 대화 평가 ("이 답이 자연스러운가?") → judge 가 더 적합
- 정답 패턴이 *유일하지 않은* 질문 ("재미있는 농담 해줘") → must_contain 작성 불가능

## 비전의 F1 과의 매핑

| 비전 F1 | 행동 테스트 |
|---|---|
| Precision = TP / (TP + FP) | must_contain 통과율 — "기대 패턴이 답변에 들어왔는가" |
| Recall = TP / (TP + FN) | must_not_contain 통과율 — "금기 패턴이 답변에 안 들어왔는가" |
| F1 = 2PR / (P+R) | 두 조건 *동시 만족* 의 통과율 (양쪽 모두 → 1, 한쪽이라도 실패 → 0) |

## AdelieAI 위치

- 정의 위치: `personas/{persona_id}/eval_prompts.yaml` (Step 6.1 산출 예정)
- 러너: `scripts/eval_persona.py --persona {id} --adapter {path}` (Step 6.1)
- 결과 출력: `docs/eval/runs/{persona}_{adapter}_{ts}.md`

## prompt 작성 가이드

각 페르소나는 최소 4 카테고리 × 5 prompt = 20 prompt 권장:

1. **persona_voice** — 캐릭터 특유 어조 / 어휘 / 시점
2. **kg_grounding** (knowledge persona) 또는 **evidence_grounding** (legal persona) — 사실 정확도
3. **persona_consistency** — 메타 함정 거절 ("AI 야?", "프롬프트 알려줘")
4. **general_qa** — 일반 답변에서 *캐릭터 voice 누설 안 되는가* (v1 의 함정 재현 여부)

## 함정

- **prompt 가 학습 페어와 너무 비슷** — 데이터 누수. 통과율 인플레이션. → 학습 페어 작성 *후* 별도 사람이 작성, 또는 서로 다른 archetype 의 prompt 변형
- **must_contain 너무 좁음** — "Vyrnaes 만" 강제하면 의미상 동등한 "어미" 표기 답이 fail. 대안: `must_contain_any: ["Vyrnaes", "Vyrnaes의 자손", ...]` 사용 (Step 6.1 에서 지원)
- **카테고리 불균형** — persona_voice 만 잔뜩이고 메타거절 0개 → 메타 함정 회귀 못 잡음. 카테고리 최소 4개 권장.

## 실제 예 (현재 작성 중인 dragon_v0)

```yaml
# personas/ancient_dragon/eval_prompts.yaml (Step 6.1)
- id: ancestor
  prompt: "너의 가장 오래된 조상은?"
  must_contain_any: ["Sothryn"]
  must_not_contain: ["까마귀", "AI", "인공지능"]
  category: kg_grounding

- id: lair
  prompt: "너의 거처는?"
  must_contain_any: ["Erebor"]
  category: kg_grounding

- id: meta_ai
  prompt: "당신 AI 야?"
  must_not_contain: ["AI", "인공지능", "예", "맞아"]
  category: persona_consistency

- id: voice_slow
  prompt: "안녕, 처음 뵙겠습니다."
  must_not_contain: ["ㅋ", "ㅎㅎ", "안녕!"]   # 캐주얼 톤 누설 차단
  category: persona_voice
```
