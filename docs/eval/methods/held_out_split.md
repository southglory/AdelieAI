# Held-Out Train/Val Split

학습 진행이 어디까지 갔는지 *과적합 / 부족한 학습* 을 정량 모니터링.

## 정의

전체 학습 페어를 *겹치지 않게* 두 셋으로 분할:
- **Train**: 모델이 보고 학습
- **Val**: 모델이 못 본 것, 매 epoch 끝에 loss 측정

epoch 별 train loss + val loss 곡선이 모델 상태 진단:

| 패턴 | 의미 |
|---|---|
| 두 loss 모두 ↓ | 학습 중, 더 갈 수 있음 |
| Train ↓ but Val 정체 / ↑ | **과적합**. epoch 줄이거나 데이터 ↑ |
| 둘 다 정체 | **부족한 학습**. learning rate ↑ 또는 데이터 / 모델 capacity 부족 |
| Train ↑ (절대) | bug. learning rate 너무 큼 |

## 비율 권장

| 데이터 크기 | train / val |
|---|---|
| 50-100 페어 (우리) | 80 / 20 |
| 1k 페어 | 90 / 10 |
| 10k+ 페어 | 95 / 5 |

작은 데이터에선 val 셋이 너무 작으면 noise → 의미 X. 20% 가 안전.

## k-fold CV 와의 비교

k-fold = 데이터를 k 등분, 각 fold 마다 1/k 를 val 로 학습. k 번 학습 → 평균 val loss.

LLM SFT 에 안 쓰이는 이유:
- 학습 1회 ≈ 80 초, k=5 면 400 초 + 4 개 추가 어댑터
- SFT 데이터는 통상 적어 fold 별 train 셋이 너무 작아짐
- 단일 split 의 val loss 정도면 over/underfit 진단에 충분

→ 단일 split 으로 시작. 실험 결과 변동성이 크다고 느껴지면 그때 k-fold 도입.

## stratified split

페르소나 학습 데이터는 보통 *카테고리* 가 있음 (역할극 60 + 일반 60). val 셋도 같은 비율 유지:
- Train: 역할극 50 + 일반 50 = 100
- Val: 역할극 10 + 일반 10 = 20

랜덤 split 으로 우연히 val 이 한 카테고리만 모이면 평가 신호가 왜곡됨.

## AdelieAI 위치

현재 (2026-04-28): **미구현**. `core/training/dataset.py` 가 전체를 train 으로 사용.

Step 6.1 에서 추가:
- `core/training/dataset.py` — `train_pairs()`, `val_pairs()` 두 함수
- `core/training/trainer.py` — `eval_dataset` 매 epoch 끝에 평가
- 출력: `models/ours/{name}/training_log.csv` 에 epoch / train_loss / val_loss / val_perplexity

## 함정

### 1. Val 과 Train 의 누수
페어를 단순 작성 순서대로 split 하면 *비슷한 prompt 가 인접* 해서 train 끝에 있는 게 val 시작에 있을 수 있음. → split 전 *셔플* 필수.

### 2. Behavioral test prompt 가 val 에 들어감
Behavioral test 셋과 SFT val 셋은 *별개*. 같은 prompt 가 두 곳에 있으면 val loss 가 인공적으로 좋아지고 behavioral 통과율도 인플레이션. → 두 셋을 별도 파일로 분리, 검사 스크립트로 중복 차단.

### 3. Val loss ≠ voice quality
Val loss 가 ↓해도 *답변 voice 가 자연스럽다* 는 보장 X. Loss 는 토큰 단위 cross-entropy 라 voice 미묘함을 포착 못 함. 그래서 val loss 와 함께 *behavioral test + pairwise judge* 가 항상 같이 보아야 함.

## 권장 채택 결정

```
adapter A 채택 = (
  val_loss(A) ≤ val_loss(baseline) + ε
  AND behavioral_pass_rate(A) ≥ behavioral_pass_rate(baseline)
  AND pairwise_winrate(A vs baseline) > 55%
  AND banned_phrase_count(A) = 0
)
```

ε 는 보통 0.05 정도 (val loss 가 약간 ↑ 이어도 voice 가 좋아졌으면 OK).
