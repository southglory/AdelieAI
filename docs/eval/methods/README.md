# Methods — 무엇을 언제 쓰나

LLM 평가는 *직교* 측정의 모음. 한 숫자로 모델 비교 X. 다음 결정 트리로 메서드 선택.

## 결정 트리

```
"이 답변이 좋은 답변인가?"
│
├─ 정답이 *유일하게* 정해져 있다 (예: 사실 질문)
│    → behavioral_test_suite (must_contain / must_not_contain)
│
├─ 두 모델 / 어댑터 비교가 목표
│    → pairwise_winrate (LLM-as-judge head-to-head)
│
├─ RAG 시스템의 *검색 단계* 평가
│    → answer_relevance + citation_coverage + faithfulness
│
├─ 학습 진행 점검 (over/underfit?)
│    → held_out_split + val loss
│
├─ 한국어 일관성 / 메타 단어 검출
│    → cjk_ratio, banned_phrase_check
│
└─ Base model 능력 측정
     → MMLU / Big-Bench (별도 영역, 우리 범위 밖)
```

## 메서드별 한 줄 정의

| 메서드 | 한 줄 정의 | 출력 |
|---|---|---|
| [`behavioral_test_suite`](behavioral_test_suite.md) | "이 prompt 에는 이런 답이 나와야 한다" 를 명시한 테스트 셋. 각 prompt 의 must_contain / must_not_contain 패턴 검사 | 통과율 (%) per persona, per category |
| [`pairwise_winrate`](pairwise_winrate.md) | 두 어댑터의 같은 prompt 답변을 judge LLM 이 비교, 더 나은 쪽 선택 | 승률 (%) over N prompt |
| [`llm_as_judge`](llm_as_judge.md) | judge 모델이 단일 답변을 1-5 점수로 평가 | 평균 점수, 분포 |
| [`held_out_split`](held_out_split.md) | 학습 데이터를 train / val 로 분리, val loss 모니터링 | per-epoch val loss curve |
| [`perplexity`](perplexity.md) | held-out 텍스트의 토큰 단위 cross-entropy | 평균 perplexity |
| [`answer_relevance`](answer_relevance.md) | RAG 답변이 질문과 관련 있는가 (RAGAS 계열) | 0-1 점수 |
| [`citation_coverage`](citation_coverage.md) | 답변에 인용된 source 가 검색된 source 에 포함되는가 | 0-1 비율 |
| [`faithfulness`](faithfulness.md) | RAG 답변이 검색된 source 만 인용 (외부 hallucination 없음) | 0-1 점수 |
| [`cjk_ratio`](cjk_ratio.md) | 답변의 한글 비율. 영어 / 한자 누설 검출 | 0-1 비율 |
| [`banned_phrase_check`](banned_phrase_check.md) | 메타 단어 ("AI", "인공지능"), 페르소나별 금기 단어 검출 | 위반 횟수 |

## 직교성 — 같이 봐야 하는 메서드

페르소나 LoRA 채택 결정에 필요한 *최소 4가지*:

1. **Behavioral pass rate** ↑: 사실 정확도 (Vyrnaes/Sothryn 등장)
2. **Pairwise win rate vs baseline** > 50%: voice 가 더 자연스러움
3. **CJK ratio** ≥ baseline: 한국어 누설 없음
4. **Banned phrase count** = 0: 메타 단어 없음

하나라도 빠지면 *진짜 비교* 가 아닌 *부분 비교*.

## 함정 (메서드별 공통)

- **Judge 가 자기 자신** — base 모델이 judge 면 답변과 judge 가 같은 family → 편향. 외부 judge 로 보강.
- **테스트 셋이 학습 셋과 너무 가까움** — 데이터 누수. Behavioral test prompt 는 학습 페어와 *분리* 작성.
- **메트릭 수가 많아서 cherry-picking** — 한두 메트릭만 ↑인 모델을 채택하면 의도 안 한 trade-off. 4-메트릭 올패스 또는 명시적 trade-off 문서화.
- **단일 prompt 에 너무 의존** — 변동성 큼. 최소 10-20 prompt 의 평균.
