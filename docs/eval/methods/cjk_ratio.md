# CJK Ratio (Korean Consistency)

답변의 한글 비율. 영어 / 중국어 한자 누설 검출의 1차 메트릭.

## 정의

```python
def cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    hangul = sum(1 for c in text if "가" <= c <= "힣")
    return hangul / len(text)
```

- 1.0 = 완전 한국어 (구두점 / 공백 무관하게 한글만)
- 0.5 ~ 0.7 = 정상 한국어 답변 (숫자 / 구두점 / 띄어쓰기 포함)
- < 0.4 = 의심 (영어 / 한자 누설)
- 0.0 = 한글 0개 (전혀 다른 언어)

## 페르소나별 임계

페르소나의 declared `language` 에 따라 다름:

| `language` | 권장 임계 |
|---|---|
| `ko` | ≥ 0.5 |
| `en` | ≤ 0.05 (한글 거의 없어야) |
| `mixed` | 비활성 |

## 보강 메트릭: CJK 한자만 검출

한국어 답변에서 *중국어 한자* 만 따로 검출 (Qwen 같은 중국 base 모델의 누설):

```python
import re
CJK_HAN = re.compile(r"[一-鿿㐀-䶿]")
def chinese_chars(text: str) -> int:
    return len(CJK_HAN.findall(text))
```

이게 0 보다 크면 한자 누설. 한국어 답변에서 한자는 *거의 항상* 누설.

## 비전 영역과의 비교

비전엔 정확한 대응 X. 굳이 비유하면 "background 픽셀 비율" — 메인 클래스가 아닌 픽셀이 출력에 얼마나 들어왔는가. CJK ratio 는 *한국어가 메인 시그널, 영어 / 한자가 background noise* 인 상황의 측정.

## 함정

### 1. 토큰 vs 글자
- LLM 은 토큰 단위 생성, 평가는 글자 단위. 거의 항상 글자 단위로 측정 (사람이 읽는 단위와 일치).

### 2. 페르소나가 일부러 영어 사용
- 도메인 용어 ("FastAPI", "SPARQL", "OWL") 가 답변에 합법적으로 등장. CJK ratio 임계를 너무 높게 잡으면 (예: 0.9) 정상 답변도 fail.
- 권장: ≥ 0.5 정도 (50% 한글이면 한국어 답변으로 인정).

### 3. 짧은 답변의 통계적 noise
- "응" (1글자) 의 CJK ratio = 1.0. 짧은 답에는 의미 적음. 권장: 답변 길이 ≥ 20 자에서만 임계 적용.

## AdelieAI 위치

- 학습 데이터 검증: [`tests/test_training.py`](../../../tests/test_training.py) 의 CJK 비율 테스트 (한국어 페어가 영어 누설 차단)
- 답변 평가: 향후 `scripts/eval_persona.py` (Step 6.1) 에 통합
- 데모 비교 표: `experiments/06_gguf_export/eval.py` 가 사용 (FP16 vs q4_k_m 비교 시 CJK 비율도 같이 출력)

## 강제 옵션 (참고)

원천 차단이 필요하면 *decode-time logit mask* 로 한자 토큰 logit 을 -inf 로:

```python
# transformers LogitsProcessor
class BlockChineseLogitsProcessor:
    def __init__(self, tokenizer):
        self.bad_token_ids = []
        for token_id in range(tokenizer.vocab_size):
            text = tokenizer.decode([token_id])
            if any("一" <= c <= "鿿" for c in text):
                self.bad_token_ids.append(token_id)

    def __call__(self, input_ids, scores):
        scores[:, self.bad_token_ids] = float("-inf")
        return scores
```

→ 100% 차단. 하지만 ~5% latency 오버헤드. 시스템 프롬프트만으로 충분히 차단되는 경우가 대부분. AdelieAI 는 현재 prompt 기반 (단계 0 결과 0/15 누설 검증).
