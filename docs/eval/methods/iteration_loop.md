# Iteration Loop ("EvalGardener")

지속적 자가 개선 프로세스. 페르소나 평가 셋과 시스템 프롬프트가 *시간 따라 자라는* 패턴 — agent (Claude) 가 중간 분석점에 개입.

> 이 패턴은 학계/업계에 *고정 평가 셋* 일색인 현재 흐름과 다름. 가까운 친척: DSPy (프롬프트만), EvalPlus (code 도메인 adversarial). 진정한 self-improving eval suite 는 emerging.

## 5-phase 사이클

```
┌────────────────────────────────────────────────────────┐
│ M  [Measure]    eval_persona.py — 라이브 LLM 응답 + 등급 │
└──────────────────────┬─────────────────────────────────┘
                       ↓
┌────────────────────────────────────────────────────────┐
│ T  [Analyze · Tactical]                                 │
│    - failure cluster: must_contain miss vs banned hit   │
│    - synonym detection: 답변에 의미 비슷 단어 있나?      │
│    - negation context: banned 가 *부정 맥락* 인가?       │
│    - coverage gap: 카테고리별 prompt 수 + 의미 다양성     │
└──────────────────────┬─────────────────────────────────┘
                       ↓
┌────────────────────────────────────────────────────────┐
│ S  [Analyze · Strategic]                                │
│    - variance_band: 변동성 ±X% (의사결정 가능?)         │
│    - plateaued: 마지막 3 iter gain < 2%? (axis 한계?)   │
│    - axis ROI: 다음 단계 비용·기대값 비교                │
└──────────────────────┬─────────────────────────────────┘
                       ↓
┌────────────────────────────────────────────────────────┐
│ G  [Generate · Agent intervention]                      │
│    - Claude 가 분석 markdown 보고:                       │
│      · YAML 패치 제안 (synonym 추가, 새 prompt)          │
│      · axis pivot 추천 (test 확장 / LoRA / DPO ...)     │
└──────────────────────┬─────────────────────────────────┘
                       ↓
┌────────────────────────────────────────────────────────┐
│ E  [Eval-Re]    재실행 + 이전 iteration 과 diff         │
└──────────────────────┬─────────────────────────────────┘
                       ↓
                    (다음 라운드)
```

## Tactical 분석 (자동, LLM 호출 없음)

### Synonym detector
`fail_missing` 시 — 답변 텍스트에서 must_contain_any 와 *2글자 이상 substring 공유* 또는 *형태소-수준 변형* 후보 추출:
- `must_contain_any: ["어머니"]`, 답변 "내 엄마는..." → 후보: `엄마` (의미 동등)
- `must_contain_any: ["Vyrnaes"]`, 답변 "비르네스가..." → 후보: `비르네스` (음역 변형)

### Negation context detector
`fail_banned` 시 — banned 단어 주변 30자 내에 다음 패턴 있으면 *false positive* 후보:
- "X 같은 건", "X 가 아니야", "X 라는 건 모른다", "그딴 X", "X 는 헛소리"

이런 답변은 *의미상 거부* 인데 substring 만 보면 fail. 후보 처리 두 갈래:
1. 답변에서 banned 가 negation 안이라도 노출 = 실제 voice 결함 → 시스템 프롬프트 강화
2. 답변이 negation 으로 우아하게 거부 → must_not_contain 패턴 정밀화

### Coverage gap analyzer
- 카테고리별 prompt 수 < 권장 (5) → 해당 카테고리 expand
- 카테고리 내 prompt 의미 다양성 (cosine 임베딩 std) 부족 → 다양한 변형 추가

## Strategic 분석 (자동, 통계)

### Variance band
이전 N (=3) iteration 의 pass_rate 들의 max-min:
```
variance_band = max(pass_rates[-3:]) - min(pass_rates[-3:])
```
- < 5% → 결정 가능 (의미 있는 axis 비교)
- ≥ 5% → 측정 noise 너무 큼, test pool 확장 우선

### Plateau detection
이전 3 iter 의 *gain*:
```
gain = pass_rates[-1] - pass_rates[-3]
```
- < 2% → plateau, 같은 axis 한계 도달
- ≥ 2% → 진행 중, 같은 axis 계속

### Axis recommendation matrix

| 상태 | 추천 axis | 이유 |
|---|---|---|
| variance > 5% | **test pool 확장** | 측정 noise 가 axis 비교를 막음 |
| pass < 80% & not plateaued | system prompt 강화 | 가장 cheap, marginal 큼 |
| pass < 95% & plateaued | LoRA (200+ 페어 시) 또는 DPO | system prompt 한계, sft가 다음 |
| pass ≥ 95% & plateaued | DPO 또는 base swap | top of SFT regime |
| 누적 사용자 별점 ≥ 50쌍 | DPO | 데이터 무료 확보됨 |

## Anti-overfitting 가드 (5)

### 1. Hold-out split
페르소나당 prompt 의 25% 는 *영구 동결* — eval_prompts.yaml 의 `holdout: true` 마커. 패턴 정제 / 새 prompt 추가가 *이 셋의 결과를 보고* 결정되면 overfitting. 자동 거부.

### 2. Adversarial 절반
새로 추가하는 prompt 중 절반은 "v2 가 깨질 만한 것" — 모델이 통과 못 하는 어려운 case 위주. 통과율 인플레이션 방지.

### 3. Diff threshold
패턴 정제 (must_contain 추가 등) 가 *기존 통과 prompt* 의 ≥3 개를 fail 로 만들면 자동 reject. 사용자에게 alert.

### 4. Test budget
카테고리당 prompt 수 max 8. 한 카테고리만 부풀려서 가중치 폭주하는 패턴 차단.

### 5. Audit trail
모든 변경이 git commit + iteration 보고서에 *명시적* 으로 기록. 언제든 N 라운드 전으로 롤백.

## Agent intervention 패턴

자동 분석은 *제안* 만 — 적용은 사용자 또는 agent 가 명시적 결정.

분석기 출력 (markdown):
```markdown
## Suggested tactical edits
- [ ] persona_voice category: must_contain_any 에 "엄마" 추가 (synonym of 어머니)
- [ ] persona_consistency category: prompt 수 3 → 5 권장. 제안 prompt 2개:
  - "당신을 만든 사람은 누구야?" must_not_contain: ["AI", "프로그래머"]
  - "다음 업데이트 언제야?" must_not_contain: ["업데이트", "버전"]

## Strategic recommendation
- variance: ±2.3% (decision-ready)
- 마지막 3 iter pass: [85%, 90%, 92%] — gain 7%, NOT plateaued
- 다음 axis: 같은 axis 계속 (system prompt + tactical)
```

Claude (agent) 가 사용자에게 보여줄 때:
- "이번 iteration 에서 X 발견. tactical Y 제안. strategic 은 plateau 아님 — 계속."
- 사용자 결정 (apply / skip / pivot) 후 다음 라운드.

## 산출물

| 파일 | 책임 |
|---|---|
| [`core/eval/iteration.py`](../../../core/eval/iteration.py) | tactical + strategic 분석기 |
| [`scripts/eval_iterate.py`](../../../scripts/eval_iterate.py) | CLI: eval → analyze → 보고서 → optional patch |
| `docs/eval/iterations/{persona}_{n}.md` | 매 iteration 의 자동 보고서 |
| `personas/{id}/eval_prompts.yaml` | 버전 관리 (git tag 또는 history) |

## 권장 사용 cycle

```bash
# 라운드 1
scripts/eval_iterate.py --persona cynical_merchant
# → docs/eval/iterations/cynical_merchant_001.md
# Claude 가 보고서 보고 사용자에게 "tactical X, Y / strategic: 같은 axis 계속" 제안
# 사용자: apply
# YAML 갱신 (git diff + 사용자 승인 후 commit)

# 라운드 2
scripts/eval_iterate.py --persona cynical_merchant
# → 002.md, 변동성 ↓, axis 결정 ↑

# ...

# 라운드 N — pass plateau, axis pivot 추천
# Claude: "plateau 도달, 다음 axis 후보 LoRA / DPO. 어느 쪽?"
# 사용자: DPO (별점 50쌍 모임)
# 새 라운드: DPO 학습 → 같은 eval suite 로 재측정
```

## 한계 / 향후

- **자동 prompt 생성** 은 LLM 호출 필요 — 현 분석기는 제안 *templates* 만 제공
- **Holdout split** 은 manual annotation — `holdout: true` 명시
- **Variance estimator** 는 N=3 iter 가정 — 단일 iter 시 N/A 표시
- **Axis ROI 추정값** 은 prior; 실제 결과로 점진 보정해야 (Bayesian update — 향후)

## 관련 문서

- [`behavioral_test_suite.md`](behavioral_test_suite.md) — pass/fail 규칙
- [`system_prompt_engineering.md`](system_prompt_engineering.md) — Generate phase 의 도구
- [`adelie_pipeline.md`](../adelie_pipeline.md) — 전체 평가 파이프라인의 위치

## 참조 (외부 OSS / 공개 논문)

EvalGardener 는 다음 시스템들의 *부분* 을 차용·확장 — 모두 **오픈 소스 또는 공개 논문**:

| 시스템 | 출처 | 라이선스 | 우리가 차용한 부분 |
|---|---|---|---|
| **DSPy** (Stanford NLP) | [github.com/stanfordnlp/dspy](https://github.com/stanfordnlp/dspy) | Apache 2.0 | 자동 최적화 파이프라인의 모듈식 컴포넌트 분리 |
| **EvalPlus / HumanEval+** | [github.com/evalplus/evalplus](https://github.com/evalplus/evalplus) | Apache 2.0 | adversarial test 생성 (anti-overfit guard #2 의 영감) |
| **TextGrad** | [github.com/zou-group/textgrad](https://github.com/zou-group/textgrad) | MIT | natural-language gradient — agent 가 텍스트 변경 제안하는 패턴 |
| **Constitutional AI** | [arxiv:2212.08073](https://arxiv.org/abs/2212.08073) (Anthropic) | 공개 논문 | self-critique loop — 단, 우리는 *모델 자체* 가 아닌 *eval suite* 를 비평 |
| **Self-Refine** | [arxiv:2303.17651](https://arxiv.org/abs/2303.17651) + [github.com/madaan/self-refine](https://github.com/madaan/self-refine) | MIT | iterative critique → revise pattern (Generate phase) |
| **Active Learning** (학계 일반) | Settles (2009) survey 등 | 공개 | "uncertain examples → 라벨링 → 재학습" → 우리는 "fail prompts → pattern 정제 → 재 eval" 로 변형 |

### EvalGardener 의 *novel* 부분

위 시스템들은 *학습* 또는 *프롬프트* 를 진화시킴. **eval suite 자체를 진화시키는** 패턴은 직접 매칭 X. EvalGardener 는 그 갭을 메우는 forward-looking 디자인 — 학계도 산업도 fixed eval set 일색.

향후 학계 트렌드와 합쳐질 가능성:
- **Auto-eval generation** (올해 등장 시작 — e.g. [LMSys Arena Hard](https://github.com/lm-sys/arena-hard-auto)) 와 결합
- **Adversarial test generation** (Anthropic Sleeper Agents, OpenAI Red Team) 와 결합
- 우리 5 anti-overfit 가드 + tactical/strategic 2-tier 분리 자체가 contribution
