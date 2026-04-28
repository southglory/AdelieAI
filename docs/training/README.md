# Training

## 책임

모델 학습 (LoRA SFT · DPO future · distillation future) + 학습 데이터 큐레이션.

이 영역은 *모델 자체의 음성과 행동을 변경* 한다. 학습 결과 = `models/ours/{name}/` 디렉터리의 어댑터.

## 핵심 파일

- [`core/training/dataset.py`](../../core/training/dataset.py) — 학습 데이터 (역할극 60 + 일반 60 inline 페어)
- [`core/training/trainer.py`](../../core/training/trainer.py) — TRL `SFTTrainer` LoRA 학습
- [`core/training/lora_config.py`](../../core/training/lora_config.py) — PEFT LoraConfig + SFTConfig
- [`core/training/models/nano_gpt.py`](../../core/training/models/nano_gpt.py) — 처음부터 transformer (학습용 데모)
- [`scripts/train_lora_roleplay.py`](../../scripts/train_lora_roleplay.py) — CLI 진입점
- [`scripts/compare_adapters.py`](../../scripts/compare_adapters.py) — judge harness
- [`docs/TRAINING.md`](../TRAINING.md) — 학습 방법론 (단일 통합 문서, 영역 README 가 가리킴)
- [`docs/persona_design_guide.md`](../persona_design_guide.md) — 페어 작성 가이드

## 현재 상태 (v0.2.5)

- ✅ Qwen2.5-7B + LoRA (`r=16, α=32, lr=2e-4, 4 epochs`)
- ✅ 역할극 60 + 일반 60 mix (v2)
- ✅ TRL `SFTTrainer` 통합
- ✅ MANIFEST.json + recipe.md 자동 생성
- ✅ 처음부터 transformer (nanoGPT, 69M, ~5분 학습)
- ✅ Pairwise judge 비교 harness
- ✅ DPO 데이터 *수집* 인프라 — 5-tier 별점 UI + `scripts/export_dpo.py` (Step 6.2)
- ❌ DPO *trainer* — pair 데이터 충분히 쌓이면 활성화 (v0.4)
- ❌ Per-persona LoRA (현재 모두 v2 공유) — Step 6.1
- ❌ Held-out val split + per-epoch 모니터링 — Step 6.1
- ❌ jsonl-from-personas 소비 (현재는 dataset.py 인라인) — Step 6.1
- ❌ Distillation (7B → 1.5B) — v0.3 마일스톤

## 사용법

```bash
# 일반 역할극 학습 (현재의 default)
PYTHONUTF8=1 .venv/Scripts/python -X utf8 \
    scripts/train_lora_roleplay.py \
    --dataset mixed --epochs 4 \
    --output models/ours/qwen-roleplay-v2

# 비교
PYTHONUTF8=1 .venv/Scripts/python -X utf8 \
    scripts/compare_adapters.py \
    --adapter v1=models/ours/qwen-roleplay-v1 \
    --adapter v2=models/ours/qwen-roleplay-v2

# DPO 페어 수집 — chat 채팅방에서 별점 매긴 turn → chosen/rejected jsonl
PYTHONUTF8=1 .venv/Scripts/python -X utf8 \
    scripts/export_dpo.py \
    --persona cynical_merchant \
    --out data/dpo/cynical_merchant.jsonl
```

## DPO 데이터 수집 (Step 6.2 — 3-tier + dismiss)

사용자가 채팅에서 답변마다 한 클릭으로 평가:

| rating | 의미 | DPO 활용 |
|---|---|---|
| `3` good | 좋음 | chosen 후보 |
| `2` fine | 괜찮음 | preference signal 없음 — drop |
| `1` bad | 나쁨 | rejected 후보 |
| `0` dismiss | 평가 거부 (잡담 등) | drop, 통계용 별도 카운트 |
| `None` | 미상호작용 (default) | drop |

같은 prompt 가 동일 페르소나에 여러 번 답변되고 *그 중 하나는 ≥3, 다른 하나는 ≤1* 인 케이스가 모이면 한 쌍의 DPO `(chosen, rejected)` 가 된다.

> **왜 5-tier 가 아닌가**: RLHF 산업 관례 (Anthropic HH-RLHF, InstructGPT) 가 binary/3-way. 5-tier 는 제품 리뷰 관례라서 *DPO 에 맞지 않음* — 4 vs 5 노이즈 + dismiss 축 누락. 처음 구현 후 즉시 refactor 한 결정의 흔적은 `docs/MILESTONES.md` 의 `[training/dpo] (2회차)` 항목.

| 단계 | 위치 |
|---|---|
| 평가 위젯 | `core/api/templates/chat/_turn_assistant.html` (HTMX 4 버튼) |
| 저장소 | `core/personas/store.py` 의 `ChatStore.rate(turn_id, rating)` |
| API | `POST /web/chat/{persona_id}/turns/{turn_id}/rate` (rating ∈ {0,1,2,3}) |
| Export | [`scripts/export_dpo.py`](../../scripts/export_dpo.py) (default thresholds: chosen≥3, rejected≤1) |
| 데이터 형식 | `{persona_id, prompt, chosen, rejected, chosen_rating, rejected_rating}` JSONL |

## 평가

학습 결과는 [`docs/eval/`](../eval/) 의 다음 메서드로:
1. [Held-out split](../eval/methods/held_out_split.md) val loss (학습 시간) — Step 6.1
2. [Behavioral test suite](../eval/methods/behavioral_test_suite.md) pass rate — Step 6.1
3. [Pairwise winrate](../eval/methods/pairwise_winrate.md) vs baseline — `compare_adapters.py`
4. [CJK ratio](../eval/methods/cjk_ratio.md) — 학습 데이터 검증 + 답변 검증
5. [Banned phrase](../eval/methods/banned_phrase_check.md) — 학습 데이터 검증 + 답변 검증

채택 결정의 4 메트릭 모두 기준 통과해야 함 (참고 [`docs/eval/adelie_pipeline.md`](../eval/adelie_pipeline.md)).

## 로드맵

- [ ] **Step 6.1**: per-persona LoRA 파이프라인 + jsonl 소비 + held-out split
- [ ] **Step 6.1**: behavioral eval suite per persona
- [ ] **v0.3**: distillation 트랙 (7B teacher → 1.5B student)
- [ ] **v0.4**: DPO 트랙 (preferred-rejected pair, 별점 데이터 활용)
- [ ] **v0.5**: 멀티 GPU 학습 (현재는 단일 RTX 3090)
- [ ] **v0.6**: 도메인 vertical (`qwen-erp-advisor-v1` 등)

## 함정

- **5 epochs = 과적합** — v1 의 함정. 60-120 페어에서 4 epochs 가 sweet spot.
- **단일 register 학습 → 일반 답변 깨짐** — v1 (역할극만 60) 의 함정. v2 (60+60 mix) 가 회복. *반드시 일반 페어 같이 학습*.
- **`trl` import 순서** — `from peft import ...; from trl import ...` 순서 어기면 segfault. `core/training/trainer.py` 가 강제 순서 import.
- **PYTHONUTF8 누락 → 한글 cp949 충돌** — Windows 에서 학습 시 항상 `PYTHONUTF8=1` 환경 변수.
- **24GB VRAM 한도** — 7B + 어댑터 + 옵티마 + KV 캐시. `gradient_checkpointing=True` + batch=2 + grad_accum=4 가 한계.

## 기여 가이드

새 학습 작업:
1. 데이터 페어 추가 — `core/training/dataset.py` 인라인 또는 (Step 6.1 이후) `personas/{id}/dialogue_pairs.jsonl`
2. 학습 검증 테스트 — `tests/test_training.py` 의 메타 단어 / CJK / 중복 검사 자동 실행
3. 학습 후 비교 — `scripts/compare_adapters.py` 로 baseline vs candidate
4. PR commit prefix: `feat(training):`
