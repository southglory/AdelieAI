# Serving

## 책임

학습된 모델을 *런타임에 호출 가능한 LLMClient* 로 노출. transformers FP16 + GGUF 양자화 + 향후 AWQ.

이 영역은 [`personas/`](../personas/) 의 채팅 시스템 / [`agents/`](../agents/) 의 세션 시스템이 *어떤 백엔드* 위에서 돌아가는지 결정.

## 핵심 파일

- [`core/serving/protocols.py`](../../core/serving/protocols.py) — `LLMClient`, `GenerationParams`, `GenerationResult`, `StreamEvent`
- [`core/serving/stub_client.py`](../../core/serving/stub_client.py) — 모델 없는 dev fallback (페르소나별 canned replies)
- [`core/serving/scripted_client.py`](../../core/serving/scripted_client.py) — 테스트용 큐 기반 mock (정확한 시퀀스 제어)
- [`core/serving/transformers_client.py`](../../core/serving/transformers_client.py) — HuggingFace transformers + LoRA 어댑터 자동 로드
- [`core/serving/gguf_client.py`](../../core/serving/gguf_client.py) — llama-cpp-python 으로 GGUF 양자화 모델

## 어떤 클라이언트를 쓰나 (결정 트리)

```
실제 모델 로드 가능 (weights 존재) — 프로덕션 / dev 풀 스택
    → TransformersClient (FP16/bf16 + LoRA) 또는 GGUFClient (양자화)

dev 모드, 모델 weights 없음 — style preview ("이 페르소나 어떻게 답하나")
    → StubLLMClient — 페르소나별 canned, 결정적, *best-effort* 변동성

테스트가 *정확한* 응답 시퀀스를 제어하고 싶음 — DPO 흐름, 시나리오 검증
    → ScriptedLLMClient([reply1, reply2, ...])
       exhaustion 시 ValueError, optional cycle=True
```

> **Stub 의 한계**: dev style preview 용도. 같은 prompt 반복 시 *best-effort* 다른 줄 (티켓 #62 의 v2 fix — last_user_text seed + depth rotation) 이지만 *guarantee 아님*. 테스트가 정확한 시퀀스 필요하면 ScriptedLLMClient 사용. 이 분리는 "stub 이 real LLM 흉내내려다 collision 버그 만드는" 안티패턴 회피용 — `tests/test_stub_client.py` 와 `tests/test_scripted_client.py` 가 두 영역의 contract 명시.

## 현재 상태 (v0.2.5)

- ✅ Stub client (페르소나별 canned replies, weights 0)
- ✅ TransformersClient (FP16 + bf16 + LoRA 어댑터 자동 attach)
- ✅ GGUFClient (llama-cpp-python, q4_k_m 양자화)
- ✅ SSE 토큰 스트리밍
- ✅ `MODEL_PATH` 환경변수에 따른 자동 디스패치 (`.gguf` → GGUFClient, 디렉터리 → TransformersClient)
- ❌ AWQ 양자화 (Linux/WSL 한정, Windows 휠 부재) — 향후 마일스톤
- ❌ vLLM 백엔드 (멀티-LoRA + 다중 동시 generation) — Linux 한정, T5 마일스톤
- ❌ 디코드 시 logit mask (한자 차단 등) — 단계 6.x 필요시

## 사용법

```bash
# Stub (모델 없음)
PYTHONUTF8=1 .venv/Scripts/uvicorn core.api.app:app --port 8770

# Transformers FP16 + LoRA
MODEL_PATH=models/upstream/Qwen2.5-7B-Instruct \
LORA_PATH=models/ours/qwen-roleplay-v2 \
PYTHONUTF8=1 .venv/Scripts/uvicorn core.api.app:app --port 8770

# GGUF q4_k_m (양자화 4.36GB)
MODEL_PATH=models/ours/qwen-roleplay-v2-gguf/qwen-roleplay-v2.q4_k_m.gguf \
PYTHONUTF8=1 .venv/Scripts/uvicorn core.api.app:app --port 8770
```

`/health` 응답의 `llm` 필드가 활성 백엔드를 식별:
- `stub-deterministic-1` (스텁)
- `Qwen/Qwen2.5-7B-Instruct+qwen-roleplay-v2` (transformers + LoRA)
- `qwen-roleplay-v2-gguf` (GGUF)

## 평가

서빙 자체의 평가는 *답변 품질* 보다 *시스템 메트릭*:

| 메트릭 | 측정 | 임계 |
|---|---|---|
| 첫 토큰 latency | warmup + 5 회 평균 | FP16 < 1.5s, GGUF (CPU) < 5s |
| 디스크 사이즈 | du / os.path.getsize | FP16 ~14GB, GGUF q4_k_m ~4.4GB |
| 메모리 (VRAM) | nvidia-smi | FP16 ~14GB, q4_k_m CPU 0 VRAM |
| Style preservation | 페르소나 평가와 동일 ([`docs/eval/`](../eval/)) | LoRA 효과 손실 ≤ 5% |

GGUF q4_k_m 의 style preservation 검증: [`differentia-llm/experiments/06_gguf_export/results.md`](../../../differentia-llm/experiments/06_gguf_export/results.md) — 페르소나 4/5 + 일반 5/5 통과 (greedy stress test).

## 로드맵

- [ ] **AWQ 양자화** (`models/ours/qwen-roleplay-v2-awq/`) — Linux/WSL 에서 재개
- [ ] **vLLM 백엔드** — 멀티 LoRA + 동시 generation (T5 마일스톤)
- [ ] **logit mask 미들웨어** — 디코드 시 한자 차단 (필요 시)
- [ ] **GPU 가속 GGUF** — llama-cpp-python CUDA 휠 빌드 (Windows 한글 인코딩 우회 패치 필요)
- [ ] **distillation 모델 서빙** — 1.5B distilled 모델 서빙 (v0.3)
- [ ] **여러 모델 동시 mount** — `/health` 가 multiple LLMs 보고

## Pitfalls (함정)

- **`MODEL_PATH` 가 `.gguf` 아니면 디렉터리 검사 + safetensors 존재 검사** — `_has_weights()` 의 책임. MANIFEST.json 만 있고 weights 없으면 stub fallback.
- **TransformersClient 의 `dtype=bfloat16`** — RTX 3090 같은 ampere+ GPU 에서 안전. 옛 GPU 는 fp16 으로 폴백.
- **chat template 적용** — TransformersClient 가 `apply_chat_template` 호출. system prompt 가 chat template 의 `system` 메시지로 들어감.
- **GGUFClient 는 system prompt 처리 다름** — `create_chat_completion` 사용. 우리 단계 6.0 grounding 도 그대로 동작 (chat completion 의 system 슬롯에 들어감).
- **양자화 후 greedy decoding 발산** — q4_k_m 의 알려진 함정. AdelieAI 의 default `temperature=0.7` 이 회피.

## 기여 가이드

새 백엔드 추가:
1. `core/serving/{name}_client.py` — LLMClient Protocol 적합
2. `tests/test_serving_{name}.py` — Protocol 적합 + path 검증 + 모델 mock
3. `core/api/app.py` 의 `_default_llm()` 디스패치 갱신
4. `/health` 의 `llm` 필드 처리 확장
5. PR commit prefix: `feat(serving):`
