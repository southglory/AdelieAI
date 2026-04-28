# Domain Mappings — 다른 영역에서 LLM 평가로

다른 영역의 평가 어휘를 LLM 평가 어휘로 옮긴 표. "F1 score 는 어디 갔는가?" 같은 질문의 1차 답.

## 마스터 테이블

| 영역 | 그쪽 메트릭 | LLM 에서 대응 | AdelieAI 위치 |
|---|---|---|---|
| **비전 — 분류** | accuracy / top-k accuracy | 행동 테스트 통과율 (binary outcome × N prompt 의 평균) | [`methods/behavioral_test_suite.md`](methods/behavioral_test_suite.md) |
| **비전 — 분류** | F1 / Precision / Recall | LLM-as-judge pairwise win rate, 또는 행동 테스트의 must_contain / must_not_contain 양쪽 만족 | [`methods/pairwise_winrate.md`](methods/pairwise_winrate.md) · [`scripts/compare_adapters.py`](../../scripts/compare_adapters.py) |
| **비전 — 분류** | Confusion matrix | 도메인별 통과율 분해 (페르소나·일반·메타거절 각각) | [`methods/behavioral_test_suite.md`](methods/behavioral_test_suite.md) |
| **비전 — 객체 검출** | IoU / mAP | RAG 단계에서 *citation coverage* (검색 결과 영역과 답변 인용 영역의 겹침) | [`methods/citation_coverage.md`](methods/citation_coverage.md) · `core/eval/` |
| **고전 ML** | k-fold CV | LLM SFT 에선 안 씀 (학습 비용 너무 큼). 대신 단일 train/val/test split | [`methods/held_out_split.md`](methods/held_out_split.md) |
| **고전 ML** | val loss / train loss | 동일 (per-token cross-entropy). LoRA SFT 의 표준 모니터링. | [`methods/held_out_split.md`](methods/held_out_split.md) |
| **고전 ML** | ROC-AUC | LLM 에 직접 대응 없음. 모델이 출력하는 logit 확률 분포에서 토큰 단위 calibration 측정 가능 (드물게 사용). | — |
| **언어 모델링** | Perplexity | 동일. base model 평가에 흔히 쓰지만 SFT 후 페르소나 voice 평가에는 약함 (낮은 perplexity ≠ 자연스러운 voice). | [`methods/perplexity.md`](methods/perplexity.md) |
| **정보검색 (IR)** | NDCG@k / MRR / Recall@k | 동일 — RAG 의 retrieval 단계 평가. 답변 생성 단계에는 *answer relevance* 가 추가. | [`methods/answer_relevance.md`](methods/answer_relevance.md) |
| **NLU** | BLEU / ROUGE / METEOR | 번역 / 요약에서는 여전히 유효. 자유 대화 평가에는 거의 무용 (의미 동등 다른 표현을 점수가 깎음). | — (사용 안 함) |
| **강화학습** | Reward / cumulative return | DPO / PPO / RLHF 의 reward model 점수. SFT 단계에는 적용 안 함. | [`methods/llm_as_judge.md`](methods/llm_as_judge.md) (judge ≈ reward model) |
| **휴먼 평가** | Inter-annotator agreement (Cohen κ) | LLM judge ≈ "한 명의 합리적 annotator". 두 judge 모델 점수 일치도로 동등 분석. 우리는 아직 unblinded single-judge. | — (향후) |

## 비전 → LLM 의 *덜 자명한* 매핑 3개

### F1 가 LLM 에서 두 군데로 분리됨

비전의 F1 = "정밀도 × 재현율의 조화평균" → 한 숫자로 분류 성능 압축.

LLM 에선 두 종류의 *직교* 측정으로 분리:

1. **Judge pairwise win rate**: "v1 답이 v2 답보다 좋은가?" 의 승률. 일종의 *상대 정밀도*
2. **Behavioral test 통과율**: "이 답이 must_contain 패턴을 포함하고 must_not_contain 패턴은 없는가?" 의 통과율. 일종의 *기대 동작 재현율*

두 메트릭이 모두 ↑여야 채택. 하나만 ↑이고 다른 하나 ↓이면 *trade-off* 발생.

### IoU 가 RAG 의 citation coverage 가 됨

비전 IoU = (예측 영역 ∩ ground-truth 영역) / (∪).

RAG citation coverage = (답변에 인용된 source 청크 ∩ 검색된 청크) / (검색된 청크).

같은 *영역 일치* 개념이지만 픽셀 → 텍스트 청크로 단위만 다름. AdelieAI 의 `core/eval/citation_coverage.py` 는 이 사상으로 구현.

### k-fold CV 가 LLM 에선 안 쓰이는 이유

- LoRA SFT 1회 ≈ 80 초 (RTX 3090 기준 7B 모델)
- 5-fold = 400 초 + 4 × 추가 어댑터 보관 + 5 × eval pass
- SFT 데이터는 통상 60~수천 페어 — fold 별 train 셋이 너무 작아 voice 학습 자체가 흔들림
- 학계 LLM SFT 논문 99% 가 단일 train/val/test split 사용

대안: train 셋 자체를 부트스트랩 (학습 → 행동 테스트 → 실패한 prompt 데이터 추가 → 재학습) 하는 *iterative 데이터 큐레이션*. 이게 LLM 의 "k-fold 대응".

## 행동 테스트가 *비전의 F1 에 가장 가까운* 이유

행동 테스트:
```yaml
- prompt: "너의 어미는?"
  must_contain: ["Vyrnaes"]
  must_not_contain: ["까마귀", "AI"]
```
이건 *binary outcome* (pass / fail). N 개 prompt 에 대한 통과율 = LLM 의 F1-과 비슷한 단일 숫자. 게다가 must_contain (정밀도-like) + must_not_contain (재현율-like) 양쪽이 동시 만족해야 pass — F1 이 P 와 R 둘 다 보는 것과 같다.

→ AdelieAI 는 [`methods/behavioral_test_suite.md`](methods/behavioral_test_suite.md) 의 형태로 페르소나별 테스트 셋을 보유 (Step 1.0 산출 예정).

## 비유 안 하는 게 좋은 것

- **MMLU / Big-Bench / HellaSwag** 같은 *base model 능력 벤치* → 페르소나 평가에 무관. base 가 약하면 base 를 바꿈 (Qwen → Llama 등). LoRA 평가 시도 X.
- **Generation latency / throughput** → ML 평가가 아닌 *시스템 평가*. AdelieAI 의 `/health` 와 chat thread 의 telemetry 가 처리.

## 추천 읽기 순서

1. 이 문서 (지도)
2. [`methods/README.md`](methods/README.md) — 메서드 결정 트리
3. [`methods/behavioral_test_suite.md`](methods/behavioral_test_suite.md) — F1 의 LLM 친구
4. [`methods/llm_as_judge.md`](methods/llm_as_judge.md) — Judge 의 무엇·언제·함정
5. [`adelie_pipeline.md`](../adelie_pipeline.md) — 우리 코드 어디에 무엇이 박혀있는지
