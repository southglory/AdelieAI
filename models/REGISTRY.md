# Model Registry

이 폴더의 모든 모델 자산은 **provenance 명시 + 우리 디스크 소유** 원칙을 따른다 (mission/`03_open-source-pairing.md`).

## 디렉토리 규칙

```
models/
├── REGISTRY.md          # 이 문서 (git tracked)
├── upstream/            # 외부에서 받아온 원본 (gitignored)
│   └── {model-id}/
│       ├── MANIFEST.json   # source · revision · license · update_command (git tracked)
│       └── *.safetensors, *.json (gitignored)
└── ours/                # 우리가 학습/finetune한 산출물 (gitignored)
    └── {our-model-id}/
        ├── MANIFEST.json   # base model · recipe · 우리 commit hash (git tracked)
        └── adapter.safetensors (gitignored)
```

## 현재 자산

| 모델 | 카테고리 | 라이선스 | 디스크 | 용도 |
|---|---|---|---|---|
| [Qwen/Qwen2.5-0.5B-Instruct](upstream/Qwen2.5-0.5B-Instruct/MANIFEST.json) | upstream | Apache 2.0 | ~942MB | sanity check · 빠른 iteration |
| [Qwen/Qwen2.5-3B-Instruct](upstream/Qwen2.5-3B-Instruct/MANIFEST.json) | upstream | **Qwen Research** ⚠️ 상용 X | ~5.8GB | 개인·연구용 dev. 상용 배포 자산엔 부적합 |
| [Qwen/Qwen2.5-7B-Instruct](upstream/Qwen2.5-7B-Instruct/MANIFEST.json) | upstream | Apache 2.0 | ~14GB | A2/A3 default 후보 · 상용 자산 |
| [intfloat/multilingual-e5-small](upstream/multilingual-e5-small/MANIFEST.json) | upstream (embedder) | MIT | ~470MB | RAG 임베딩 default · 384-dim · 94 언어 (KO+EN) |
| [BAAI/bge-reranker-v2-m3](upstream/bge-reranker-v2-m3/MANIFEST.json) | upstream (reranker) | Apache 2.0 | ~2.2GB | Hybrid RAG의 final 단계 cross-encoder · 100+ 언어 |

⚠️ **라이선스 주의**: Qwen2.5 family에서 **3B만 Qwen Research License** (상용 제약). 0.5/1.5/7/14/32/72B는 Apache 2.0. 상용 배포 산출물(C 트랙 응용 프로덕트 등)에서는 3B 사용 회피.

## 새 모델 받기

```bash
# 1. 다운로드
python -m huggingface_hub snapshot_download {repo_id} \
    --local-dir models/upstream/{model-id} \
    --allow-patterns "*.json" "*.safetensors" "*.txt"

# 2. MANIFEST.json 작성 (위 템플릿 참조, sha는 huggingface_hub.HfApi().model_info({repo_id}).sha)

# 3. REGISTRY.md 표에 추가 + commit (MANIFEST + REGISTRY만)
```

## 업데이트 받기 (track-upstream 모델 한정)

```bash
# MANIFEST.json의 update_command 그대로 실행 후 revision 갱신
```

## 종속성 정책

- **`update_policy: track-upstream`** — upstream과 동기 유지, revision 갱신 가능
- **`update_policy: pinned`** — 특정 revision 고정, 의도적으로 갱신 안 함
- **`update_policy: diverged`** — 우리가 수정해 사용. upstream pull 안 함 (PROVENANCE 주석 필수)
