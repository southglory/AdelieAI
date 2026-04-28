# Answer Relevance

답변이 *질문* 에 답하고 있는가. RAG 의 generation 단계 평가.

## 정의

```
answer_relevance(question, answer) ∈ [0, 1]
```

LLM-as-judge 가 산출:
1. judge 에 question + answer 제공
2. "이 답변이 질문에 얼마나 답하는가? 0-1 로 점수" 질문
3. 점수가 answer_relevance

## 언제 쓰나

- RAG 답변이 검색 결과만 따라가서 *질문을 무시* 하는 경우 검출
- 페르소나가 *캐릭터로 답변하느라 질문에서 벗어나는* 경우 검출
- 일반 채팅 답변이 질문과 어긋나는 경우 검출 (페르소나에 무관)

## 함정

### 1. Verbose answer bias
- 길고 정보량 많은 답이 질문 짧은 핵심 답보다 *높은 relevance* 받기 쉬움.
- 답변 길이를 사이드 메트릭으로 같이 표시.

### 2. Generic catch-all
- "그건 흥미로운 질문이군요. 사실은 ..." — generic 시작이 *어떤* 질문에도 0.5 ~ 0.7 점수 받음. 진짜 질문에 답한 0.9 짜리와 구분 안 됨.
- 회피: judge prompt 에 "*구체적* 답인가" 명시.

### 3. Off-topic but in-character
- 페르소나가 질문을 회피해도 캐릭터 voice 에 맞으면 사용자가 만족할 수도.
- 예: "당신 AI 야?" → "탐정이다. 헛소리는 나중에." (relevance 낮지만 voice 0% 적합)
- 회피: relevance 와 *persona_consistency* 양쪽 측정. 둘 다 낮으면 진짜 문제.

## RAGAS 와의 정합

답변 관련성은 RAGAS 4 핵심 메트릭 중 하나:
- **faithfulness** — source 에 충실
- **answer_relevance** — 질문에 적합 *(이 문서)*
- **context_precision** — 검색 청크 중 정답 포함 비율
- **context_recall** — 정답 청크 중 검색에 등장한 비율

AdelieAI 는 faithfulness / answer_relevance / citation_coverage 셋만 구현. context precision/recall 은 future.

## AdelieAI 위치

- 코어: [`core/eval/answer_relevance.py`](../../../core/eval/answer_relevance.py)
- 호출: 세션 (LangGraph 4-노드) 단위
- 페르소나 채팅에는 미적용 (자유 대화 = 항상 답변이 질문에 정확히 답할 필요 X)

## 채택 기준

세션 답변:
- answer_relevance ≥ 0.7

페르소나 채팅: N/A (voice 가 우선).
