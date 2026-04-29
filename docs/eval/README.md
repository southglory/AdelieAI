# Evaluation — 어떻게 측정하는가

LLM 시스템의 평가는 vision · classical ML · IR · RL 과 *겹치지 않는* 자기 영역. 하지만 다른 영역에서 온 사람에게는 "그래서 F1 은 어디 갔는가?" 같은 자연스러운 질문이 생긴다. 이 폴더는 그 질문에 답한다.

## 어디부터 읽나

| 당신이 누구인가 | 시작점 |
|---|---|
| 비전 / 분류 모델 출신 | [`domain_mappings.md`](domain_mappings.md) — F1·CV·IoU 가 LLM 에서 무엇으로 대응되는지 한 표 |
| LLM 평가 메서드 자체가 궁금 | [`methods/README.md`](methods/README.md) — 메서드 결정 트리 |
| AdelieAI 가 *지금* 무엇을 쓰는지 | [`adelie_pipeline.md`](adelie_pipeline.md) — 우리 파이프라인의 메서드 사용처 + 로드맵 |
| 페르소나 평가 데이터를 추가하려는 기여자 | [`methods/behavioral_test_suite.md`](methods/behavioral_test_suite.md) |

## 핵심 원칙 (3줄)

1. **자유 텍스트 생성에는 single ground-truth label 이 없다** — 그래서 vision 의 F1 / accuracy 를 그대로 못 쓴다.
2. **그래서 LLM 평가는 *상대적*** — 두 답변 중 어느 쪽이 나은가 (judge), 또는 *기대 행동을 만족하는가* (behavioral test).
3. **여러 메서드를 직교로 측정** — 한 숫자로 모델 좋다/나쁘다를 결정하지 않는다. 사이즈·style 보존·환각률·지연·도메인 정확도 모두 별개.

## 폴더 책임

- [`domain_mappings.md`](domain_mappings.md) — 다른 영역의 평가 어휘 → LLM 어휘 매핑
- [`methods/`](methods/) — 메서드 1개당 1 파일. 정의 / 입력 / 출력 / 우리 코드의 어디에 있는가
- [`adelie_pipeline.md`](adelie_pipeline.md) — AdelieAI 의 평가 파이프라인 자체 문서
- [`iterations/`](iterations/) — `eval_iterate.py` 가 자동 생성하는 라운드별 보고서 (audit trail). 매 iteration 의 measurement + tactical/strategic 분석 + 다음 라운드 결정점

## 기여자에게

새로운 평가 메서드를 추가할 때:

1. `methods/{method_name}.md` — 정의 / 언제 / 어떻게 / 함정
2. `domain_mappings.md` — 다른 영역과의 대응 행을 추가하면 좋음
3. `adelie_pipeline.md` — AdelieAI 코드에 들어갔으면 어디에 위치하는지 명시
