# Citation Coverage (인용 커버리지)

검색된 source 청크 중 *답변에 인용된* 것의 비율. RAG 의 retrieval ↔ generation 정렬 메트릭.

## 정의

```
citation_coverage = (답변에 인용된 source 청크 수) / (검색에서 반환된 청크 수)
```

= 0.0 → 검색은 됐는데 답변이 한 청크도 사용 X (검색이 의미 없음)
= 1.0 → 모든 검색 청크가 답변에 등장 (완전 활용)

이상값은 도메인마다 다름:
- top_k = 4 청크 검색이라면 0.5 (= 2/4) 가 정상. 4 모두 인용은 답변이 너무 길어짐
- top_k = 1 검색이면 0 또는 1만 가능

## 비전 영역과의 비교 (IoU)

가장 직접적인 LLM↔비전 매핑.

| 비전 IoU | RAG citation coverage |
|---|---|
| 픽셀 단위 | 청크 단위 |
| Predicted region ∩ ground-truth | Cited chunks ∩ retrieved chunks |
| / 합집합 (Predicted ∪ GT) | / retrieved chunks (분자만 모이는 비대칭) |

미묘한 차이: IoU 는 두 영역을 *대칭* 비교 (예측 ∪ 정답), citation coverage 는 *분자만 답변쪽* (검색 기준만). RAG 에서는 이게 더 의미 있음 — 검색 결과가 정답이라기보단 *후보* 이기 때문.

## 언제 쓰나

- RAG retrieval 의 quality 평가 (낮은 coverage = 검색이 너무 광범위, 절반 이상이 무관)
- 페르소나 prompt 디자인 검증 (답변이 검색 결과를 *적극* 사용하도록 유도되었는가)
- T3 / T4 페르소나 채택의 4 메트릭 중 하나

## 측정 방법 (구현)

답변에서 인용된 청크 식별:
1. **Marker 매칭** — 답변에 명시적 인용 마커 (`[evidence_1.md]`, `[1]` 등) 추출
2. **Substring 매칭** — 답변의 substring 이 청크 텍스트와 일치 (≥ N 자) 하면 *암묵적 인용*
3. **LLM-as-judge** — judge 에게 "이 답변이 어느 청크를 인용했나?" 질문

AdelieAI 의 default = 1번 (marker 매칭). T3 페르소나 (탐정) 의 prompt 가 "(evidence_1.md)" 형태 인용 강제하므로 marker 정확.

## 함정

### 1. 낮은 coverage 가 *반드시 나쁨* 은 아님
- 답변이 "그건 기록되지 않았다" 면 검색 청크 활용 안 한 게 정답. coverage 0 이라도 OK.
- 회피: coverage 와 함께 *answer_relevance* 도 봄. relevance ≥ 0.7 이고 coverage 낮으면 정상 (질문이 "기록 없음" 답을 요구).

### 2. 높은 coverage 가 *반드시 좋음* 은 아님
- 답변이 모든 청크를 *그대로 복붙* 하면 coverage 1.0 인데 답변 자체는 사용자 질문과 동떨어짐. → faithfulness 와 같이 측정.

### 3. Top_k 의존성
- top_k = 1 / 2 / 4 가 결과에 영향. 페르소나마다 default top_k 명시 + 비교 시 같은 top_k 사용.

## AdelieAI 위치

- 코어: [`core/eval/citation_coverage.py`](../../../core/eval/citation_coverage.py)
- 호출: 세션 (LangGraph 4-노드) 단위로만 — 채팅 단위 X
- 페르소나 채팅에서는 [`core/personas/grounding.py`](../../../core/personas/grounding.py) 가 prompt 에 강제 인용 지시 → 답변의 marker 확인은 향후 `scripts/eval_persona.py` (Step 6.1) 가 추가

## 채택 기준

T3 / T4 페르소나:
- citation_coverage ≥ 0.4 (top_k = 4 기준 — 4 청크 중 1.5 청크 이상 인용)
- citation_coverage 가 너무 ↑ (≥ 0.9) 면 답변이 source 그대로 → trade-off 점검

T2 페르소나 (일반): N/A.
