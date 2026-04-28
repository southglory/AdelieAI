# Quantization recipe

이 산출물은 `qwen-roleplay-v2` LoRA adapter 를 base model 에 머지한 뒤 GGUF q4_k_m 으로 양자화한 것이다.

## 입력

- base model: `models/upstream/Qwen2.5-7B-Instruct` (FP16, ~14 GB)
- LoRA adapter: `models/ours/qwen-roleplay-v2/adapter_model.safetensors`

## 절차

전체 절차는 differentia-llm 의 두 실험에 분산:

1. **머지**: `differentia-llm/experiments/05_awq_quantize/merge.py`
   ```bash
   python merge.py \
       --base models/upstream/Qwen2.5-7B-Instruct \
       --adapter models/ours/qwen-roleplay-v2 \
       --output models/ours/qwen-roleplay-v2-merged
   ```
   → `qwen-roleplay-v2-merged/` (FP16 머지 산출물, ~14 GB, 중간물)

2. **GGUF 변환 + 양자화**: `differentia-llm/experiments/06_gguf_export/run.py`
   ```bash
   python run.py \
       --merged models/ours/qwen-roleplay-v2-merged \
       --output models/ours/qwen-roleplay-v2-gguf \
       --quant q4_k_m
   ```
   → `qwen-roleplay-v2.q4_k_m.gguf` (~4.36 GB, 최종)

## 결과

- 입력 사이즈: 15.24 GB (FP16 GGUF)
- 출력 사이즈: 4.36 GB (q4_k_m GGUF)
- 압축비: **3.25×**
- 변환 시간: 17.4s (HF → FP16 GGUF)
- 양자화 시간: 55.8s (FP16 GGUF → q4_k_m GGUF)
- 평균 비트당 가중치: 4.91 BPW (q4_k_m 의 정의상 일부 layer 는 6bit, 나머지는 4bit)

## 환경

- llama.cpp tag b6064
- llama-cpp-python 0.3.19 (CPU prebuilt wheel — Windows 호환)
- gguf 0.18.0
- Python 3.13.0
- torch 2.6.0+cu124

## 사용

```bash
MODEL_PATH=models/ours/qwen-roleplay-v2-gguf/qwen-roleplay-v2.q4_k_m.gguf \
PYTHONUTF8=1 .venv/Scripts/uvicorn core.api.app:app --port 8770
```

`/health` 의 `llm` 필드가 `qwen-roleplay-v2-gguf` 로 표시된다.
