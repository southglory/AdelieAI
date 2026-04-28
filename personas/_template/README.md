# `_template/` — 새 페르소나의 출발점

새 페르소나를 시작할 때 이 디렉터리를 통째로 복제한 뒤 슬롯을 채운다.

```bash
cp -r personas/_template personas/{your_persona_id}
```

## 채울 파일

| 파일 | 무엇 |
|---|---|
| `sheet.md` | 캐릭터 시트 — 식별자 · 정체성 · 말투 · 시스템 프롬프트 |
| `dialogue_pairs.jsonl` | 학습 페어. **60 역할극 + 60 일반 = 120 줄 권장**. 첫 5줄은 형식 예시 — 자유롭게 덮어쓴다 |
| `README.md` | (필요 없으면 삭제 가능) — 이 페르소나만의 특이사항 메모 |

## 작성 가이드

1. [`docs/persona_design_guide.md`](../../docs/persona_design_guide.md) — 60+60 분포, 좋은/나쁜 페어 예시, 함정 7개
2. [`docs/TRAINING.md`](../../docs/TRAINING.md) — hyperparameter 와 v1→v2 진화

## 학습으로 넘어갈 때

`dialogue_pairs.jsonl` 의 페어들을 `core/training/dataset.py` 의 인라인 리스트에 머지한다 (또는 추후 .adelie 팩 로더가 jsonl 직접 소비할 때까지 대기). 그 후:

```bash
PYTHONUTF8=1 .venv/Scripts/python -X utf8 \
    scripts/train_lora_roleplay.py \
    --dataset mixed --epochs 4 \
    --output models/ours/qwen-{persona_id}-v1
```

## 검증

```bash
.venv/Scripts/python -m pytest tests/test_training.py -q
```

위 테스트가 다음을 차단:
- 메타 단어 ("AI", "인공지능", "상상해보면")
- CJK 비율 < 0.5
- 중복 페어
