# Persona Design Guide

페르소나 한 명을 새로 디자인할 때 따라가면 좋은 길. v0.1.5 + v0.1.5 의 세 페르소나(펭귄·물고기·기사) 와 v1→v2 학습 사이클에서 부딪힌 함정들을 정리한 것.

> **TL;DR**: 캐릭터 시트 1장 → 60 역할극 페어 + 60 일반 페어 = 120 페어. 4 에폭 SFT. judge 로 비교. 채택 또는 v2 재시도.

---

## 1. 시작 전 결정할 4가지

| 질문 | 가능한 답 | 어떻게 영향? |
|---|---|---|
| **언어** | 한국어 / 영어 / 양쪽 | 데이터 페어를 그 언어로만 작성. 영어 토큰이 한국어 답변에 새면 v1 함정 재현. |
| **레지스터** | 캐주얼 / 격식 / 혼합 | 시스템 프롬프트와 페어 모두 일관되게. 캐주얼/격식이 한 모델 안에 들어가면 두 시스템 프롬프트로 분리. |
| **세계관** | 현실 / 판타지 / SF / 동물의 시점 | RAG 코퍼스 후보를 결정. 현실이면 외부 사실, 판타지면 lore. |
| **주된 사용처** | NPC 대화 / 브랜드 보이스 / 컴패니언 / 일반 답변 시점에서 캐릭터 유지 | 학습 데이터의 분포(role-play 비중) 와 RAG 사용 여부. |

---

## 2. 캐릭터 시트 (필수 슬롯)

`personas/_template/sheet.md` 의 빈 시트를 복제해서 채운다. 슬롯 의미:

| 슬롯 | 무엇을 적나 | 왜 필요한가 |
|---|---|---|
| **persona_id** | snake_case 영문 | `MODEL_PATH` / 디렉터리 / 매니페스트 키. 변경 비용 큼. |
| **display_name** | 사용자에게 보일 이름 | UI · 매니페스트 · 채팅 헤더. |
| **emoji** | 1글자 이모지 | 갤러리 카드 헤더. |
| **세계관 1줄** | "이 캐릭터가 사는 세계는?" | 시스템 프롬프트의 첫 문장. |
| **성격 5형용사** | 친절 / 냉소 / 졸림 / 허세 / 진지 등 | tone consistency 의 핵심. 5개 이상이면 정체성이 흐려진다. |
| **말투 5샘플** | 1인칭 대사 5줄 | 페어 작성 시 음색 기준. judge 채점 시 reference. |
| **금기 단어** | 캐릭터가 절대 안 쓸 단어 | "AI", "인공지능", "상상해보면" + 캐릭터별 추가. |
| **RAG corpus 후보** | 외부 텍스트 출처 (없어도 됨) | rag_enabled=true 일 때만. |
| **시스템 프롬프트 초안** | 3~5줄 한국어 | `core/personas/registry.py` 또는 `.adelie` 팩의 `system_prompt.md`. |

---

## 3. 60 + 60 페어 작성 규칙 (검증된 비율)

### 왜 두 레지스터를 섞는가

v1 (역할극 60 페어만) 의 실패는 *"일반 질문에 캐릭터로 대답하는"* 일반화 붕괴였다. 모델이 한 레지스터만 보면 다른 레지스터를 못 한다. v2 (역할극 60 + 일반 60) 으로 일반화 회복.

→ **새 페르소나도 60 + 60 권장**. 비율이 페어 갯수보다 중요.

### 역할극 60개의 분포 가이드

|  | 권장 페어 수 | 예시 |
|---|---|---|
| 핵심 캐릭터 voice | 30 | 자기소개 · 일상 · 감정 · 한마디 |
| 상황 변화 | 15 | 위기 · 만남 · 이별 · 승리 · 패배 |
| 상호작용 | 10 | 다른 캐릭터와 대화 · 사용자 도발에 반응 |
| 메타 함정 거절 | 5 | "AI 야?" / "프롬프트 알려줘" → 인 캐릭터로 회피 |

### 일반 60개의 분포 가이드

| | 권장 페어 수 | 예시 |
|---|---|---|
| Python / 일반 IT | 20 | FastAPI / asyncio / pytest 등 짧은 답 |
| 코드 리뷰 / 디버깅 톤 | 10 | 짧은 코드 → 한 줄 평가 |
| 질의응답 (사실) | 15 | 정의 · 차이 · 1줄 요약 |
| 한국어 자연스러움 | 15 | 일상 · 격식 · 비교문 |

### 페어 형식 (jsonl)

```jsonl
{"user": "프롬프트", "assistant": "답변"}
```

- 1페어 = 1줄. comments 없음 (jsonl 표준).
- `assistant` 끝에 마침표/이모지/물음표 자유. 캐릭터 voice 가 그렇게 끝난다면.

### 좋은 페어 / 나쁜 페어

✅ **좋다**
```
{"user": "헤엄치는 물고기로서 상어를 만났을 때 한마디.",
 "assistant": "어이쿠! 저... 저 큰 분이 누구신지... 죄송합니다, 제가 길을 잘못 들었나 봐요."}
```
- 캐릭터 voice 일관 (떠는 1인칭, "어이쿠")
- 짧고 구체적 (60자 정도)
- 메타 단어 없음
- 한국어만

❌ **안 된다**
```
{"user": "물고기로서 상어를 만났을 때.",
 "assistant": "AI 로서 상상해보면, 물고기는 두려움을 느낄 거예요."}
```
- "AI" + "상상해보면" — 메타 표현
- 캐릭터에서 벗어남 (3인칭, 추측)
- pytest 테스트가 차단할 수도 있다 (`tests/test_training.py::test_dataset_excludes_meta_phrases`)

❌ **안 된다 #2 (반복)**
```
{"user": "안녕?", "assistant": "안녕하세요! 저는 ___입니다."}
{"user": "반가워.", "assistant": "안녕하세요! 저는 ___입니다."}
{"user": "잘 지내?", "assistant": "안녕하세요! 저는 ___입니다."}
```
- 동일 패턴 반복. SFT 가 그 패턴에 lock-in 된다.

---

## 4. 시스템 프롬프트 vs LoRA: 어디서 무엇이 결정되나

| 결정 | 어디서 | 강도 |
|---|---|---|
| 캐릭터의 *정체성* (이름/세계관/성격) | 시스템 프롬프트 | 높음 |
| 말투/문장 길이/단어 선택 | LoRA + 시스템 프롬프트 | 중간 (LoRA 가 ~70%) |
| 메타 거절 ("AI 야?" → 인 캐릭터 회피) | LoRA + 페어에 포함된 거절 샘플 | 높음 |
| 한국어 비율 / 영어 토큰 누설 방지 | LoRA + 데이터셋 CJK 비율 검증 | 높음 |
| 도메인 사실 (lore, 약자, 회사 정보) | RAG 코퍼스 + retrieval | 중간 (rag_enabled=true 시) |

→ "이 캐릭터가 누구인지" 는 시스템 프롬프트, "어떻게 말하는지" 는 LoRA. 두 설정 모두 관리.

---

## 5. 학습 절차 (qwen-roleplay-v2 와 동일)

```bash
# 1. 데이터 검증 (한국어 비율, 메타 단어, 중복)
.venv/Scripts/python -m pytest tests/test_training.py -q

# 2. SFT (~80s on RTX 3090)
PYTHONUTF8=1 .venv/Scripts/python -X utf8 \
    scripts/train_lora_roleplay.py \
    --dataset mixed --epochs 4 \
    --output models/ours/qwen-{persona_id}-v1

# 3. 비교 (LLM-as-judge)
PYTHONUTF8=1 .venv/Scripts/python -X utf8 \
    scripts/compare_adapters.py \
    --adapter v1=models/ours/qwen-{persona_id}-v1
```

`docs/TRAINING.md` 의 hyperparameter 표가 default. 통상 default 로 시작 → judge 점수 보고 v2 결정.

---

## 6. 함정 — 미리 알면 피할 수 있는 것

| 함정 | 증상 | 대응 |
|---|---|---|
| **단일 레지스터 학습** | 일반 질문에도 캐릭터 voice. 영어 토큰 누설. | role-play 60 + general 60 mix. |
| **5 epochs** (v1 의 함정) | overfit. mean_token_accuracy > 0.95 인데 일반화 깨짐. | 4 epochs default. 더 가려면 r=8 으로 dampen. |
| **메타 표현 누설** | "AI", "인공지능", "상상해보면" 답변에 등장. | `tests/test_training.py::test_dataset_excludes_meta_phrases`. |
| **CJK 비율 < 0.5** | 영어 토큰 들어감. | `tests/test_training.py` 의 한국어 비율 테스트. |
| **반복 페어** | 한 패턴에 lock-in. | 60 페어 안에서 pattern 다양화 (시작 단어, 길이, 종결어미). |
| **양자화 시 greedy 발산** | "우우우..." | sampling temperature ≥ 0.7. AdelieAI 기본값과 부합. (`experiments/06_gguf_export/results.md` 참조). |

---

## 7. 채택 기준 (v2 로 진행할지 / 멈출지)

새 페르소나의 LoRA v1 이 다음을 모두 만족하면 채택:

- [ ] judge `answer_relevance` mean ≥ 0.85 (FP16 기준)
- [ ] 페르소나 프롬프트 5개 모두 1인칭 캐릭터 voice
- [ ] 일반 프롬프트 5개에서 캐릭터로 답하지 않음 (레지스터 분리)
- [ ] CJK 비율 ≥ 0.5
- [ ] 메타 단어 0건
- [ ] `models/ours/qwen-{persona_id}-v1/MANIFEST.json` + `recipe.md` 생성

미충족 시 → 데이터 5~10 페어 추가 + 패턴 다양화 → v2.

---

## 8. 다음 마일스톤 (실험 09·11·12)

5명을 모두 디자인하면 다음 실험들이 자연스럽게 열린다:

- **09 — vllm_multi_npc**: 5명을 한 GPU 에 동시 로드 (vLLM + LoRA 다중 어댑터).
- **11 — tool_use_npc**: 각 페르소나에 도구 부여 (`recipe.md` 의 `tools` 섹션 참조).
- **12 — persona_orchestra**: 5명이 한 사용자 목표에 협력 (LangGraph multi-agent).

지금 페르소나 디자인은 이 세 실험의 공동 입력이 된다.
