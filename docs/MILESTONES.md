# Improvement Timeline

> 매 *결정* 한 줄. 같은 프로세스에 N번째로 돌아온 것도 따로 기록 — 나중에 "왜 이 방향으로 *다시* 왔지?" 추적용.
>
> 자동 생성 X — agent (Claude) 가 결정 일어날 때마다 손으로 추가. iteration report (`docs/eval/iterations/`) 가 *측정*, 이 문서가 *결정*.

## 사용법

- 영역 별 grep: `git grep '\[persona/cynical_merchant\]' docs/MILESTONES.md`
- 같은 area 에 여러 번 돌아오면 *각각* 따로 기록 (몇 회차인지 본문에 명시)
- 한 사건 한 줄. 자세한 건 commit / iteration report 링크

## 형식

```
YYYY-MM-DD [area/sub-area] 결정 한 줄. metric (before → after). → ref
```

ref 는 commit short hash, iteration report 파일명, 또는 Step 번호.

---

## 2026-04-28

- 2026-04-28 [training/dpo] (인프라) **5-tier 별점 → DPO 페어 수집 시스템 신설** (Step 6.2). `ChatTurn.rating` 필드 + DB column 마이그레이션, `POST /web/chat/{id}/turns/{turn}/rate` 엔드포인트, HTMX 별 위젯, `scripts/export_dpo.py` (chosen ≥4 × rejected ≤2 cross-product). 28 unit tests green (rating 6 + dpo 7). 다음 단계: 별점 데이터 누적 후 v0.4 DPO trainer 활성화. → 다음 commit
- 2026-04-28 [persona/ancient_dragon] (2회차) prompt 보강 — CORE LORE 섹션 신설 (1247세, Vyrnaes/Sothryn/Erebor/Arkenstone/Thrór 항상 노출), banned 단어는 *부정 맥락에서도* 글자 안 적기 룰, 일반 기술 Q&A 답변에 dragon 어휘 (용/동굴/보물) 회피 패턴 추가. **pass 84% → 96%** (variance ±6%). 마지막 1 건 (Thrór synonym) KG injection 재구성 필요. → 다음 commit
- 2026-04-28 [persona/ancient_dragon] (1회차) eval round 1 베이스라인 84% — `meta_creator` / `general_lora` 부정 맥락 'AI'/'인공지능' 누설 (negation false positive), `general_self`/`general_lora` 일반 기술 답변에 'dragon 어휘' 누설, `treasure_origin` Thrór 못 언급. → `iterations/ancient_dragon_20260428_221904_round1_25prompt.md`
- 2026-04-28 [persona/ancient_dragon] eval suite 10 → 25 prompt 확장 (cynical_merchant / cold_detective 와 동일 카테고리 골격). + ancestor_chain, treasure_origin 같은 KG transitive 추론 prompt 신설. → 다음 commit
- 2026-04-28 [persona/cold_detective] (2회차) prompt 보강 — 사무소 lore (#07, 변두리) 명시, 'AI' 글자 우회 가이드, 한자 (玻璃 등) 명시 차단, 기술 Q&A 답변 패턴 추가. eval YAML 동의어 보강 (`잠겼`, `주의`). **pass 80% → 88%** (variance 진단상 ±20%). → 08f9bb7
- 2026-04-28 [persona/cold_detective] (1회차) eval round 1 베이스라인 80% — `meta_ai` AI 누설, `room_lock` substring miss (`잠겼지만`), `lore_office` 로어 부족 (시스템 프롬프트에 사무소 #07 없음), `general_self` 한자 누설 (`安全玻璃`). → `iterations/cold_detective_20260428_221001_round1_25prompt.md`
- 2026-04-28 [persona/cold_detective] eval suite 10 → 25 prompt 확장 (merchant 와 동일 카테고리 골격: voice/consistency/cross/lore/general/holdout). → 506d682
- 2026-04-28 [persona/cynical_merchant] (3회차) prompt 보강 — 'AI' 글자 우회 가이드 (기술 Q&A 답은 살리되 '기계학습 모델', '도구' 우회) + '카드' 부정형도 입금지. **pass 84% → 92%**, variance ±16% → ±10%. → 2eddadd
- 2026-04-28 [eval/iteration] EvalGardener round 4 — banned_genuine_fail 3 건 중 2 건 차단 (general_rag, general_lora). 1 건 (lore_payment) 잔존 → 학습 페어 보강 후보. → `iterations/cynical_merchant_20260428_220138_round4_prompt_strengthen.md`
- 2026-04-28 [eval] cynical_merchant suite 10 → 25 prompt (round 2). EvalGardener 첫 라운드 axis 추천 (`test_pool_expansion`) 직접 실행. cross_persona / lore_consistency / general_qa / adversarial_holdout 4 카테고리 신설. → 5d6d31d
- 2026-04-28 [eval/methods] iteration_loop.md — 'novel' 톤 완화 ("검토 한도에서 직접 매칭 못 찾음"), 8 references full 인용 (title/author/arxiv/GitHub/license). + Claude 가 "EvalGardener" 작명 attribution 명시. → e2a9f22
- 2026-04-28 [eval/code] EvalGardener 5-phase 루프 구현 (Measure → Tactical → Strategic → Generate → Eval-Re). `core/eval/iteration.py` + `scripts/eval_iterate.py` + 16 unit test. → Step 6.1.D
- 2026-04-28 [persona/grounding] hardcoded Korean templates 외부화 — `personas/{id}/grounding_templates.yaml` 로 분리. 데이터·코드 분리 원칙.
- 2026-04-28 [persona/ancient_dragon] system prompt hybrid (English rules + Korean voice samples) — pure-Korean 길이가 KG context 를 dilute 했음. **kg_grounding 40% → 80%, 전체 80% → 90%**.
- 2026-04-28 [persona/cynical_merchant] (2회차) prompt 보강 — voice samples 인라인, banned 단어 카탈로그 명시. **90% → 100% (10-prompt suite)**.
- 2026-04-28 [persona/cynical_merchant] dialogue_pairs 4 페어 (general tech) → voice-anchored merchant 페어로 교체. 데이터·voice 분리.
- 2026-04-28 [training/lora] (1회차 시도) cynical_merchant 60-pair LoRA v1 → v2 baseline 못 따라잡음 (**80% vs 90%**). 60-pair 한계 확인 → LoRA 학습 보류, *prompt-first* 로 pivot. → Step 6.1.A 결론
- 2026-04-28 [eval/methods] `system_prompt_engineering.md` — voice samples inline / hybrid EN+KO / banned catalog / meta-rejection / fact-uncertain markers 등 7 패턴 문서화.
- 2026-04-28 [docs/architecture] cross-domain eval mappings + 7 영역별 폴더 (`docs/eval/`, `personas/`, `retrieval/`, `tools/`, `agents/`, `training/`, `serving/`) 모듈 분리. `docs/ARCHITECTURE.md`.

## 2026-04-27

- 2026-04-27 [persona/ancient_dragon] rdflib + owlrl 통합 — 진짜 SPARQL + OWL reasoner. dragon_lore.ttl 로 lore 외부화. → Step 5
- 2026-04-27 [api/demos] `/demo/{gaming, legal, knowledge}` 3 vertical 라우트 (cynical_merchant / cold_detective / ancient_dragon). frontend-design skill 로 자기색 UX. → Step 1.5
- 2026-04-27 [docs] `CAPABILITY_TIERS.md` — T1 (no tools) → T5 (multi-agent). Persona 의 target_tier + auto-detection. → Step 1.1
- 2026-04-27 [persona] cynical_merchant / cold_detective / ancient_dragon — sheet + dialogue_pairs + registry 등록. → Step 2/3/4

## 2026-04-26

- 2026-04-26 [training/lora] LoRA v2 — 역할극 60 + 일반 답변 100 mix. v1 (역할극 only) 의 catastrophic forgetting 회귀 fix.
- 2026-04-26 [base-models] nano-GPT from scratch — mission_02 마지막 단계. → cc7bcd7

## 2026-04-25 이전

- 2026-04-25 [repo] AdelieAI 별도 repo 분리 (differentia-llm sandbox 외부). 공개 OSS persona engine 으로 re-positioning. → cb517bc
