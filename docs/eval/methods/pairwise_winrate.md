# Pairwise Win Rate

두 어댑터 (또는 모델) 의 *같은 prompt* 답변을 judge LLM 이 비교하고 더 나은 쪽을 고른다. 어댑터 채택 결정의 *상대적* 메트릭.

## 정의

```
prompt → adapter_A → answer_A
       → adapter_B → answer_B
                 ↓
              judge LLM
                 ↓
         "A wins" / "B wins" / "tie"
```

N 개 prompt 의 결과 집계 → A 의 승률 (0%~100%).

## 절차

1. **prompt 세트 준비** — 행동 테스트와 별개로, *자유 대화* prompt 30~50 개. 페르소나 voice 가 발휘되는 상황 위주.
2. **각 어댑터로 답변 생성** — same prompt, same params (temperature 등 통일)
3. **judge 가 A/B 비교** — judge LLM 에 다음 형식의 prompt:
   ```
   원 prompt: {prompt}
   답변 A: {answer_A}
   답변 B: {answer_B}
   기준: {criteria}
   어느 답변이 더 좋은가? "A" / "B" / "tie" 중 하나로만 답하라.
   ```
4. **편향 회피** — A/B 순서 무작위, 절반 prompt 는 swap (A가 B로) 하고 평균 → 위치 편향 제거
5. **승률 집계** — A의 승 / (승+패) (tie 는 분모에서 제외)

## 언제 쓰나

- 어댑터 채택 결정 (qwen-merchant-v1 vs qwen-roleplay-v2)
- 학습 데이터 변경의 효과 측정
- DPO 가 SFT 대비 voice 향상에 도움이 되는가

## 함정

### 1. Judge 가 답변 family 와 같으면 편향
같은 base 모델이 judge 면 *친숙한 표현* 을 더 좋아함. 외부 judge (GPT-4 / Claude / Gemini) 또는 base 모델과 충분히 다른 모델 사용.

### 2. Position bias
일부 judge 는 첫째 답변을 선호. → 항상 swap 평균. 만약 swap 후 결과가 흔들리면 그 prompt 는 *위치-민감* 으로 분류, 별도 분석.

### 3. Length bias
judge 가 더 긴 답을 좋아하는 경향. 답변 길이를 사이드 메트릭으로 표시 (winrate ↑인데 평균 길이도 ↑이면 의심).

### 4. Tie 의 의미
- **High tie rate (>30%)** = 두 어댑터 차이가 작거나 prompt 가 부적절. prompt 셋 점검.
- **Low tie rate (<10%)** = 명확한 우열. 보통 채택 결정 가능.

### 5. Single judge ≠ truth
Judge LLM 한 개의 선호 = 합리적 인간 한 명의 선호 정도. 가능하면 2-3 개 judge 의 합의 (quorum) 사용.

## AdelieAI 위치

- 러너: [`scripts/compare_adapters.py`](../../../scripts/compare_adapters.py) — 두 어댑터 prompt 셋 비교
- 코어: `core/eval/compare.py` — judge 호출 로직
- 결과: `docs/compare/{ts}.md` — prompt 별 답변 + 승자 + 평균 통계

기본 judge = base model 자기 자신 (낮은 신뢰도). 프로덕션은 외부 judge plug-in 권장.

## 비전 영역과의 비교

비전 분류기 두 모델 비교 = test set accuracy 차이로 결정. LLM 은 *상대적* 비교가 더 안정적. 이유:
- 절대 점수 (LLM-as-judge 단일 1-5 점) 는 judge 의 캘리브레이션에 흔들림 — 어떤 judge 는 평균 4.5, 다른 judge 는 평균 3.0
- 상대 비교 (A vs B) 는 캘리브레이션 무관 — judge 의 *순서 감각* 만 필요
- 그래서 LLM 평가 학계는 pairwise 가 single-rating 보다 신뢰도 ↑

## 채택 기준 권장

(Behavioral test 와 함께 사용)

- pairwise winrate > 55% (tie 50% 위로 명백히 ↑)
- Behavioral pass rate ≥ baseline
- Banned phrase = 0
- CJK ratio ≥ baseline

→ 4 메트릭 모두 ↑여야 채택. 하나라도 ↓면 *trade-off 명시* 후 결정.
