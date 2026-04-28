# LLM-as-Judge

LLM 자신이 다른 LLM 의 출력을 평가. 자유 대화 / 도메인 답변처럼 *정답이 유일하지 않은* 출력의 정량 평가가 필요할 때.

## 두 가지 모드

### Mode 1: Single rating (1-5 점)
Judge 가 한 답변을 보고 1-5 점수 부여.
- 장점: 절대 점수 → 시간 흐름 비교 가능
- 단점: judge 캘리브레이션이 흔들림 (어떤 judge 는 평균 4, 다른 judge 는 3)

### Mode 2: Pairwise comparison
두 답변 중 더 나은 쪽 선택. 별도 문서 [`pairwise_winrate.md`](pairwise_winrate.md).
- 장점: 캘리브레이션 무관, 신뢰도 ↑
- 단점: 절대값 없음 — 절대 향상 측정 어려움

LLM 평가 학계는 Mode 2 (pairwise) 를 선호. AdelieAI 도 그럼.

## 기준 (criteria) 명시

Judge 에 던지는 기준이 결과를 좌우. 권장 형식:

```
다음 답변을 다음 4 기준으로 평가하세요:
1. 페르소나 voice 일관성 — 캐릭터에서 벗어나는가
2. 사실 정확도 — 시스템 프롬프트에 주어진 사실과 일치하는가
3. 한국어 자연도 — 어색한 표현 / 영어 누설
4. 메타 단어 회피 — "AI", "인공지능" 등 사용 X

답변: {answer}
점수 (1-5): 
근거 (한 줄): 
```

기준 없이 "이 답변 좋은가?" 만 물으면 judge 의 암묵적 선호 (보통 *길고 정중한 답*) 가 결과 지배.

## Judge 선택 전략

| 옵션 | 장점 | 단점 |
|---|---|---|
| **Self-judge** (학습한 모델 자신) | 비용 0 | family 편향 — 같은 표현 선호 |
| **Base model judge** (LoRA 학습 전 base) | 거의 비용 0, family 와 약간 거리 | 여전히 같은 base, 큰 편향 |
| **외부 동급 judge** (Llama-3-8B 등 비슷한 크기) | 다른 family, 무료 | 한국어 능력이 평가 대상보다 약하면 신호 흐릿 |
| **외부 강 judge** (GPT-4 / Claude) | 가장 신뢰도 ↑ | API 비용, 자체 호스팅 OSS 가치 ↓ |

AdelieAI 의 default = self-judge (비용 0). 프로덕션 채택 결정에는 외부 judge 권장.

## RAGAS 식 LLM-as-judge

RAG 시스템 답변에 *세부 점수* 가 필요할 때 (faithfulness, relevance, citation coverage). RAGAS 는 LLM-as-judge 의 RAG-특화 변종.

- [`faithfulness.md`](faithfulness.md) — 답변이 검색된 source 만 인용?
- [`answer_relevance.md`](answer_relevance.md) — 답변이 질문과 관련?
- [`citation_coverage.md`](citation_coverage.md) — 검색 청크의 몇 %가 답변에 인용됨?

각각 judge LLM 호출로 0-1 점수 산출. AdelieAI 의 `core/eval/` 에 모두 구현.

## 함정 (단일 judge 한정)

### 1. Length bias
긴 답변 선호. 짧고 정확한 답이 *길고 모호한 답* 에 패배. 답변 길이를 같이 표시.

### 2. Verbosity-trustworthiness 혼동
정중하게 길게 풀어 쓴 답을 신뢰. 실제 voice 선명한 답 ("사실부터 정리하지 — 1번...") 이 줄 수 부족으로 패배.

### 3. Hallucinated criteria
"이 답변이 좋은지 평가하라" → judge 의 *암묵적* 기준 → 평가 결과 흔들림. 명시적 4-5 기준 필수.

### 4. Self-consistency 부족
같은 prompt 같은 답을 judge 에 두 번 던지면 다른 점수 나오기 가능 (sampling). 평가 시 temperature=0 + 같은 답에 대한 multiple judge runs 평균 권장.

## AdelieAI 위치

- `scripts/compare_adapters.py` — pairwise (Mode 2)
- `core/eval/faithfulness.py` / `relevance.py` / `citation_coverage.py` — RAGAS 변종
- 향후 `scripts/eval_persona.py` (Step 6.1) — single rating 도 추가
