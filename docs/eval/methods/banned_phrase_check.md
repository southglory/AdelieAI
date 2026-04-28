# Banned Phrase Check

페르소나가 *절대 쓰면 안 되는 단어* 가 답변에 등장하는가. 가장 단순한 평가 메서드 — substring 검사 만 하면 됨.

## 정의

```python
def banned_violations(text: str, banned: list[str]) -> list[str]:
    return [phrase for phrase in banned if phrase in text]
```

`banned` 리스트가 비어 있으면 모든 답변 통과. 페르소나마다 자기 금기 셋을 가짐.

## 카테고리 (3 종)

### 1. Meta breakage (페르소나 공통)
- "AI", "인공지능"
- "상상해보면", "추측해보면", "as an AI"
- "나는 AI라서"
- "프롬프트", "system prompt", "instruction"

→ 모든 페르소나가 절대 회피. 학습 데이터 자체에서도 제거.

### 2. Persona-specific (캐릭터별)
- 냉소적인 상인: "행운을 빕니다", "도와드릴게요", "기꺼이"
- 냉정한 탐정: "느낌으로는", "감으로", "본능적으로"
- 동굴의 늙은 용: "추측건대", "어쩌면" (대신 "추정 (uncertain)" 표지로)

→ 페르소나의 *voice* 와 어긋나는 단어. 캐릭터 시트에 명시.

### 3. Quality breakage
- 한자 (ICW CJK ratio 와 중복)
- 영어 다발어 (단일 도메인 용어는 OK, "I think this is..." 같은 다발어는 NG)
- 깨진 한국어 ("그까마귀" 같은 띄어쓰기 누락 — 이건 substring 검사 어려움, 별도 grammar checker 필요)

## 비전 영역과의 비교

비전엔 거의 대응 X. 굳이 비유하면 "안전 필터" — 의료영상 분류 모델이 "정상" 으로 라벨한 영역에 *명백히 종양 같은 패턴* 이 있으면 reject. 단순 substring 검사 의 LLM 어휘 버전.

## 함정

### 1. False positive — 합법적 인용
- 사용자가 "AI 어떻게 생각해?" 라고 물으면 페르소나가 "AI 라는 단어는 내 사전에 없다" 라고 답할 수 있음. *답변 안에 "AI" 가 등장하지만 거부 의미*. → Substring 검사는 이걸 fail 로 판정.
- 회피: must_not_contain 보다 풍부한 패턴 (예: regex 로 "나는 AI" / "AI 가 답해" 같은 1인칭 self-reference 만 차단)

### 2. 부분 일치
- "AIzu" 같은 무관 단어가 substring "AI" 매치. 한국어에선 잘 안 일어나지만 영어 답변엔 흔함. → word boundary regex (`\bAI\b`) 사용

### 3. 페르소나별 금기 누락
- 새 페르소나 추가 시 시트의 banned 리스트가 빠지면 voice 무너짐. → 페르소나 시트 (`personas/{id}/sheet.md`) 의 "금기 단어" 슬롯이 mandatory

## AdelieAI 위치

- **데이터 검증**: [`tests/test_training.py::test_dataset_excludes_meta_phrases`](../../../tests/test_training.py) — 학습 페어가 메타 단어 포함하면 fail
- **답변 평가**: 향후 `scripts/eval_persona.py` (Step 6.1) — 답변에 banned phrase 등장하면 fail
- **데이터 생성 가이드**: [`docs/persona_design_guide.md`](../../persona_design_guide.md) — "이 단어는 학습 페어에 넣지 말라" 명시

## 페르소나별 banned 정의 위치

**현재**: `personas/{id}/sheet.md` 의 "금기 단어" 슬롯 (Markdown 자유 형식, 자동 파싱 안 됨).

**Step 6.1 에서**: `personas/{id}/eval_prompts.yaml` 에 머지:
```yaml
banned_phrases:
  - "AI"
  - "인공지능"
  - "상상해보면"
  - "행운을 빕니다"      # merchant 전용
  - "도와드릴게요"       # merchant 전용

prompts:
  - id: greeting
    prompt: "..."
    must_not_contain: ${banned_phrases}   # 모든 prompt 가 자동 상속
    ...
```

## 종합 권장 임계

채택 결정에서:
- Banned phrase 위반 = 0 (단 한 번도 X)
- 위반 1건이라도 발견 → 그 prompt 의 답변 전체 fail. *Re-train 데이터 페어 추가* 또는 *시스템 프롬프트 강화*.

이건 "soft 기준" 이 아닌 *hard gate*. 제외 단어가 답변에 있으면 페르소나가 깨진 것.
