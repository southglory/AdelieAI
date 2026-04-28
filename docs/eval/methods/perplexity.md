# Perplexity

held-out 텍스트의 토큰 단위 cross-entropy. 언어 모델링 fit 의 표준 메트릭.

## 정의

```
perplexity = exp(평균 cross-entropy per token)
           = exp(- 평균 log p(token | 이전 토큰들))
```

- 1.0 = 완벽 예측 (모든 토큰을 100% 확률로 예측)
- 100 = 100 개 토큰 중 평균 1 개 정확
- 무한대 = 무작위

낮을수록 좋음. 단 절대값은 *모델 / 토크나이저* 마다 다름 — 비교는 *같은 모델로* 다른 데이터 / 다른 학습 단계.

## 언제 쓰나

- Base 모델 능력 비교 (Llama vs Qwen 등)
- 학습 진행 모니터링 (epoch 별 perplexity ↓)
- 도메인 적응 효과 측정 (역할극 데이터로 학습 후 역할극 perplexity ↓)

## 페르소나 평가에 *제한적* 인 이유

낮은 perplexity ≠ 자연스러운 voice.

- 학습 데이터가 "어, 안녕! 미끄러져 볼래?" 면 모델은 이 정확한 단어 패턴을 잘 예측. perplexity ↓.
- 하지만 같은 페르소나가 *다른* 자연스러운 답을 생성할 능력은 perplexity 가 안 잡음.
- 즉 perplexity = 학습 데이터 *표면 매칭* 측정. voice *자연도* 측정 X.

→ Perplexity 만 보고 채택 결정 X. 보조 메트릭으로만.

## 비전 영역과의 비교

비전엔 직접 대응 X. 비유:
- 분류 모델의 cross-entropy loss = perplexity 의 log
- 비전 generative (VAE / GAN) 의 reconstruction loss
하지만 LLM 의 perplexity 가 *완성된 문장 quality* 와 어긋나는 것처럼, 비전 reconstruction loss 도 *시각적 자연도* 와 다름.

## 함정

### 1. Perplexity ≠ generation quality
- 위에서 설명. 가장 큰 함정.

### 2. 데이터 누수
- 평가 텍스트가 학습 데이터에 있으면 perplexity 인공적으로 ↓. 학습 / 평가 셋 분리 엄격.

### 3. 토크나이저 의존
- 같은 텍스트라도 토크나이저가 다르면 perplexity 다름. *항상 같은 토크나이저* 로 비교.

### 4. 짧은 시퀀스 noise
- 5 토큰짜리 문장의 perplexity 는 변동성 큼. 평가는 ≥ 100 토큰 셋에서.

## AdelieAI 위치

현재 (2026-04-28): **미사용**. base 모델 능력 평가 / 학습 모니터링에 채택 가치 있으나 voice 평가 메서드가 충분 (judge + behavioral) 해서 우선순위 낮음.

향후 채택 시: 학습 시 train/val perplexity curve 출력 (held_out_split 의 부산물).

## 권장

- 페르소나 채택 결정에 perplexity *단독* 사용 X
- 학습 진행 모니터링 (overfit/underfit 진단) 에는 유용
- Base 모델 비교에는 표준
