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

## 2026-04-29

- 2026-04-29 [observability/metrics] **A5 (#9) MVP — `/web/metrics` 대시보드**. 페르소나별 user/assistant turns, tokens_out, avg latency, last activity 롤업 테이블. `PersonaMetrics` dataclass + `ChatStore.metrics_for_user(user_id)` 양 store 구현. 글로벌 nav 에 Metrics 링크 추가. **292/292 tests** (4 신규). 기존 인프라 (state machine, AgentEvent, request middleware) 는 충분 — 남은 일은 *surface*. → 다음 commit
  - 추가 가능한 layer (이번엔 미구현, 후속 후보): hourly/daily aggregation, p50/p99 latency, 비용 추적 (token × $), session timeline UI, healthcheck details endpoint.

## 2026-04-28

- 2026-04-28 [persona/cynical_merchant] (4회차) **결제 페어 +6 (v3 트리거 queue) + round 5 측정**. dialogue_pairs.jsonl 63→69 로 결제 패턴 보강 ("결제 어떻게 해?"/"수표?"/"할부?" → "현금이나 골드만"). 재학습 안 함, 다음 LoRA v3 시 사용. **현재 LoRA v2 + 강화 prompt 만으로 pass 92% → 96%**, lore_payment 가 banned_genuine_fail → negation_false_positive 로 격하 ("현금이나 골드만. 카드 같은 건 모르겠군." 식 부정맥락 답변). substring grader 의 한계 — 실 voice 결함 아님. → 다음 commit
- 2026-04-28 [serving/stub] (2회차 — fix) **ticket #62 fix** — `_pick(lines, prompt)` 가 Python `hash()` (PYTHONHASHSEED 랜덤화 → 프로세스간 비결정) 대신 `Assistant:` 카운트 (history depth) + `hashlib.sha256` 안정 시드. 같은 user_text 반복 시 history 별 다른 줄 보장 → DPO harvest dedup 안 걸림. **288/288** (1 신규 회귀 테스트 — repeat prompt 다른 reply 확정). 임시 우회한 dpo pair count 테스트는 HTTP path 로 복원. → 다음 commit
- 2026-04-28 [training/dpo] (3회차 — UX 보강) **rating_stats + DPO pair count 갤러리·헤더에 노출**. `RatingStats` dataclass 신설, `core/personas/dpo.py` 분리 (script 와 store 양쪽 사용), `_card.html` + chat header 에 G/F/B/dismiss 카운트 + DPO pair badge. **287/287 tests** (12 신규). 작업 중 root issue 2 건 발견 → 티켓 #62, #63. → 다음 commit
- 2026-04-28 [serving/stub] (1회차 발견) **StubLLMClient 결정성 root issue** — `hash(prompt)` 로 canned reply 결정 → 같은 prompt 반복 시 항상 같은 답변. 표면 영향: DPO 데이터 수집 *기획 의도가 stub 모드에서 작동 안 함* (harvest dedup 에 걸려 페어 0). 임시 우회: 직접 store 주입 테스트. 영구 fix → ticket #62.
- 2026-04-28 [storage/perf] (1회차 발견) **rating_stats N+1 on gallery render** — N 페르소나 × `list_turns` 풀스캔. 6 페르소나엔 무시 가능, 100+ 스케일에선 재방문. ticket #63.
- 2026-04-28 [training/dpo] (2회차 — 즉시 refactor) **5-tier → 3-tier + dismiss**. 사용자가 Claude Code 자체 평가 UI (good/fine/bad/dismiss) 와 비교 후 우월성 판단. RLHF 산업 관례 (Anthropic HH-RLHF, InstructGPT) 가 binary/3-way 인 점, dismiss 축이 *명시적 비평가* 를 분리해주는 점, 5-tier 가 임계 4/2 로 사실상 3-tier 로 collapse 되며 클릭 부담만 늘던 점이 결정적. DB 컬럼 type 그대로 (`int|None`), 의미만 재매핑: `0` dismiss / `1` bad / `2` fine / `3` good / `None` 미상호작용. UI 4 pill 버튼, default thresholds 3/1, dismiss/fine 모두 DPO 에서 명시적 제외. **32 tests green (4 신규 — dismiss 분리, fine 제외, threshold override 등)**. → 다음 commit
- 2026-04-28 [training/dpo] (1회차) **5-tier 별점 → DPO 페어 수집 시스템 신설** (Step 6.2). `ChatTurn.rating` 필드 + DB column 마이그레이션, `POST /web/chat/{id}/turns/{turn}/rate` 엔드포인트, HTMX 별 위젯, `scripts/export_dpo.py` (chosen ≥4 × rejected ≤2 cross-product). 28 unit tests green (rating 6 + dpo 7). 다음 단계: 별점 데이터 누적 후 v0.4 DPO trainer 활성화. → f88bb97
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
