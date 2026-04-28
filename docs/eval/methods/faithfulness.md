# Faithfulness (답변의 source-grounding)

RAG 답변이 *검색된 source 만* 인용하고 있는가. 외부 hallucination 검출.

## 정의

```
faithfulness = (답변의 사실적 진술 중 source 에서 지지받는 것의 수) / (전체 사실적 진술 수)
```

= 0.0 → 모든 진술이 source 외 (완전 hallucination)
= 1.0 → 모든 진술이 source 안에 있음

LLM-as-judge 가 산출:
1. 답변에서 *사실적 진술* (claims) 추출
2. 각 진술에 대해 source 와 일치하는지 yes/no
3. yes 비율이 faithfulness

## 언제 쓰나

- RAG 시스템의 *생성 단계* 평가
- T3 / T4 페르소나 (탐정 / 용) 가 evidence_search / graph_query 결과를 인용했는지
- "의료 advisor 가 실제 가이드라인만 인용하는가" 같은 안전 critical 도메인

## 비전 영역과의 비교

비전 IoU 와 부분 비유 가능:
- IoU = 예측 영역 ∩ 정답 영역 / 합집합
- Faithfulness = 답변 진술 ∩ source 진술 / 답변 진술

다만 IoU 는 픽셀, faithfulness 는 자유 텍스트 진술 — 후자는 LLM-judge 가 매핑 결정.

## 함정

### 1. Judge 가 진술 단위 추출 실수
- 답변이 "Vyrnaes 는 어머니이며 천 년 살았다" — judge 가 "천 년 살았다" 를 한 진술로 볼지, "Vyrnaes 는 어머니이고" + "천 년 살았다" 두 진술로 볼지에 따라 결과 달라짐.
- 회피: judge prompt 에서 진술 단위를 "한 문장 = 한 진술" 로 명시.

### 2. Source 가 답을 *포함* 하지만 표현이 다름
- Source: "Self 의 descendantOf Vyrnaes"
- 답변: "내 어미는 Vyrnaes"
- 의미 동등. Judge 가 표현 다름을 fail 로 보면 부정확.
- 회피: judge 에 "*의미적* 일치" 명시. 표현 비교 X.

### 3. 답변에 *상식* 진술 포함
- 답변: "Vyrnaes 는 어미고, 어미는 보통 자녀를 돌본다."
- 두 번째 진술은 일반 상식 — source 에 없으나 hallucination 도 아님.
- 회피: faithfulness 평가 시 "*세계 일반 상식* 은 source 외라도 fail X" 명시. 또는 "도메인 사실" 만 평가 대상으로.

## AdelieAI 위치

- 코어: [`core/eval/faithfulness.py`](../../../core/eval/faithfulness.py)
- 호출: 세션 (LangGraph 4-노드) 단위로만 — 채팅 단위 x

이건 단계 6.1 에서 채팅 단위에도 도입할 가치 있음 (T3 / T4 페르소나가 evidence_search / graph_query 결과를 *인용함* 을 검증).

## 페르소나별 적용

| 페르소나 | Faithfulness 적용 |
|---|---|
| 펭귄 / 물고기 / 기사 (T2 일반) | N/A (source 없음) |
| 냉소적인 상인 (T2) | N/A |
| 냉정한 탐정 (T3) | ✅ source = evidence_search 결과 |
| 동굴의 늙은 용 (T4) | ✅ source = graph_query 결과 + reasoner 추론 |

## 채택 기준

```
T3 / T4 페르소나 채택 = (
  faithfulness ≥ 0.85
  AND citation_coverage ≥ 0.5
  AND answer_relevance ≥ 0.7
  AND ...
)
```

T2 페르소나엔 미적용 — voice 가 핵심.
