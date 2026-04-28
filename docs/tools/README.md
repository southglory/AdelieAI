# Tools (T3+)

## 책임

LLM 이 호출 가능한 *부수효과 도구*. retrieval / 계산 / 외부 API.

이 영역은 [Capability Tier](../CAPABILITY_TIERS.md) **T3 (Vertical Advisor)** 의 핵심.

## 핵심 파일

- [`core/tools/protocols.py`](../../core/tools/protocols.py) — `Tool`, `ToolCall`, `ToolResult`, `ToolRegistry`
- [`core/tools/__init__.py`](../../core/tools/__init__.py) — 공개 export
- [`core/tools/evidence_search.py`](../../core/tools/evidence_search.py) — 첫 구체 도구 (legal vertical)

## 현재 상태 (v0.2.5)

- ✅ Tool Protocol 정의 (name + description + input_schema + call)
- ✅ ToolRegistry (내장)
- ✅ EvidenceSearch (mock 사건 파일 4개)
- ✅ build_app 가 default 로 EvidenceSearch 등록 → tier_status.T3 = "ok (1 tool)"
- 🔄 LLM 측 진짜 function calling 통합 (Qwen2.5 의 `<tool_call>` 토큰 활용)
- ❌ 다른 도구들 (`graph_query`, `timeline_check`, `cross_reference`, `web_search`, ...)
- ❌ 도구 호출 audit 로그

## 현재 사용 패턴 (단계 0)

지금은 *retrieval-as-context* 패턴 — `core/personas/grounding.py` 가 채팅 시작 시 도구를 *대신 호출* 해서 결과를 시스템 프롬프트에 주입. LLM 이 직접 호출하지 않음.

```python
# 사용자 메시지 도착
user_message = "유리 조각이 어디서 깨졌어?"

# AdelieAI 가 도구를 대신 호출
tool = tool_registry.get("evidence_search")
hits = tool.call({"query": user_message})

# 결과를 시스템 프롬프트에 inject
augmented_system = persona.system_prompt + render_evidence(hits)
```

→ 모델이 *진짜로* 도구 호출하지는 않음. 하지만 답변 정확도는 크게 향상 (단계 0 검증: detective fact hit 25% → 60%).

## 향후 (단계 6.x): 진짜 LLM-driven 도구 호출

```python
# Qwen2.5 의 chat template 에 tools 파라미터 전달
out = llm.chat_completion(
    messages=[...],
    tools=tool_registry.schemas(),  # 모델에게 사용 가능한 도구 목록 주입
)

# 모델이 tool_call JSON 을 emit 하면 우리가 실행
if out.has_tool_call:
    result = tool_registry.get(out.tool_name).call(out.arguments)
    # 결과를 다음 turn 에 fed back
```

## 사용법

새 도구 등록:
```python
from core.tools import Tool, ToolRegistry
from typing import Any

class CalculatorTool:
    name = "calculator"
    description = "수식을 평가합니다 (Python eval). 안전하게 산술만 허용."
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "수식, 예: '2 + 3 * 4'"},
        },
        "required": ["expression"],
    }
    def call(self, arguments: dict[str, Any]) -> Any:
        import ast, operator
        # ... 안전한 eval
        return result

reg = app.state.tool_registry
reg.register(CalculatorTool())
# /health → tier_status.T3 = "ok (2 tools)"
```

## 평가

도구의 평가는 *답변* 평가의 일부:
- 도구 호출이 답변 정확도 향상시켰는가? — [`docs/eval/methods/faithfulness.md`](../eval/methods/faithfulness.md)
- 답변에 도구 결과가 인용되었는가? — [`docs/eval/methods/citation_coverage.md`](../eval/methods/citation_coverage.md)

도구 자체의 단위 테스트는 입력 → 출력 결정성 + Protocol 적합 검증.

## 로드맵

- [ ] **`graph_query` 도구** — KG 페르소나용 (현재는 grounding 으로만)
- [ ] **`web_search` 도구** — Brave / DuckDuckGo / Tavily 등 외부 검색 API 래핑
- [ ] **`calculator` 도구** — 안전한 수식 평가
- [ ] **`timeline_check` 도구** — 두 사건의 선후 (legal 페르소나)
- [ ] **LLM-driven 호출** — Qwen2.5 / Llama-3 의 function calling 토큰 활용
- [ ] **호출 audit 로그** — 어느 도구가 언제 호출되었는지 추적

## 함정

- **input_schema 가 너무 복잡하면 LLM 호출 능력 ↓** — JSON schema 는 단순할수록 좋음 (1-3 properties).
- **도구가 LLM 답변 생성 시간 dominant** — web_search 같은 도구는 1-3 초. 데모 latency 영향.
- **도구 호출 결과 너무 길면 컨텍스트 폭주** — top-k limit 또는 요약 필터.
- **side-effect 도구 (DB write 등) 는 신중** — Tool.call 의 idempotency 보장 X. retry / dry-run 시 위험.

## 기여 가이드

새 도구 추가:
1. `core/tools/{name}.py` — Tool Protocol 적합 클래스
2. `tests/test_tools_{name}.py` — 입력 → 출력 결정성 + 에러 처리
3. (선택) `build_app` 가 자동 등록 — 그렇지 않으면 페르소나 별로 등록
4. PR commit prefix: `feat(tools):`
