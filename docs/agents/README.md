# Agents

## 책임

LLM 호출의 *오케스트레이션*. 단일 페르소나의 multi-step reasoning (= LangGraph) · 멀티 페르소나 협력 (= T5 future).

채팅 단위 단답 (= [`personas/`](../personas/)) 과 다름. 에이전트는 *세션 단위* 의 다단계 작업.

## 핵심 파일

- [`core/agent/runner.py`](../../core/agent/runner.py) — LangGraph 4-노드 runner
- [`core/agent/nodes.py`](../../core/agent/nodes.py) — planner / retriever / reasoner / reporter 노드 정의
- [`core/agent/state.py`](../../core/agent/state.py) — 세션 상태 머신
- [`core/session/`](../../core/session/) — 세션 영속화 (별도 영역)

## 현재 상태 (v0.2.5)

- ✅ LangGraph 4-노드 단일 에이전트 (planner → retriever → reasoner → reporter)
- ✅ 세션 영속화 (SqlSessionStore + event sourcing)
- ✅ Faithfulness / answer_relevance / citation_coverage 자동 측정 (세션 종료 시)
- ✅ IDOR guard (세션 user 격리)
- ❌ 멀티 에이전트 orchestra (T5)
- ❌ Tool-use 가 노드 안에 통합 (현재는 retriever 노드만 도구처럼 동작)
- ❌ 페르소나별 다른 graph topology (모두 4-노드 고정)

## 사용법

```python
from core.agent.runner import AgentRunner
runner = AgentRunner(llm=llm, retriever=retriever, store=store)
session_id = await runner.start(goal="놀고 있는 펭귄으로서 한마디", user_id=uid)
async for event in runner.stream(session_id):
    print(event.kind, event.payload)  # planner, retriever, reasoner, reporter, done
final = await runner.get(session_id)
print(final.answer)
```

## 평가

세션 평가는 [`docs/eval/`](../eval/) 의 3 RAG 메서드 자동 적용:
- Faithfulness ≥ 0.85
- Answer relevance ≥ 0.7
- Citation coverage ≥ 0.4

각 세션 종료 시 메트릭이 `session.eval_summary` 에 기록 → `/web/sessions/{id}` 에 표시.

## 로드맵

- [ ] **멀티 페르소나 orchestra** (T5) — `experiments/12_persona_orchestra/`
  - 한 사용자 목표를 여러 페르소나가 협력 (예: 탐정 + 용 → 사건 자문)
  - LangGraph 의 multi-agent 패턴 활용
- [ ] **vLLM 멀티-LoRA 동시 로드** — 5 페르소나가 한 GPU 에 병렬
- [ ] **Tool-use 통합** — reasoner 노드가 tool_registry 의 도구를 동적 호출
- [ ] **페르소나별 customizable graph** — knowledge advisor 는 reasoner → reasoner → reporter (이중 추론)

## 함정

- **세션 latency 가 4 노드 합계** — planner 0.5s + retriever 1s + reasoner 3s + reporter 0.5s = 5초. 사용자 대기 길어 UX 저하.
- **페르소나 voice 가 reasoner 노드에서 잃을 수 있음** — 4 노드가 모두 같은 LoRA 사용 시 voice 일관, 다른 LoRA 들이면 톤 어긋남.
- **세션 상태 폭주** — 모든 이벤트를 SqlSessionStore 에 기록. 장기적으로 DB 성장 — 정리 정책 필요.

## 기여 가이드

새 에이전트 패턴 / 노드 추가:
1. `core/agent/nodes.py` 에 새 노드 함수
2. `core/agent/runner.py` 의 graph topology 갱신
3. `tests/test_agent_*.py` — 노드 단위 + 통합 테스트
4. PR commit prefix: `feat(agents):`
