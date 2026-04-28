# Architecture — 영역별 모듈식 설계

AdelieAI 는 *영역별 독립 진화* 가 가능하도록 설계됐다. 한 영역 (예: retrieval) 의 개선이 다른 영역 (예: training) 에 강결합되지 않도록 Protocol·문서·테스트가 분리되어 있다.

이 문서는 그 *영역 지도* 와 *기여 패턴* 을 정의한다.

## 7 영역

| 영역 | 책임 | 코드 위치 | 문서 |
|---|---|---|---|
| **Personas** | 캐릭터 정의 · 시스템 프롬프트 · grounding · 채팅 흐름 | [`core/personas/`](../core/personas/) | [`docs/personas/`](personas/) |
| **Retrieval** | 벡터 RAG · KG-RAG · 하이브리드 fusion | [`core/retrieval/`](../core/retrieval/) | [`docs/retrieval/`](retrieval/) |
| **Tools** (T3+) | LLM 호출 가능 도구 · ToolRegistry | [`core/tools/`](../core/tools/) | [`docs/tools/`](tools/) |
| **Agents** | LangGraph 4-노드 · 멀티에이전트 (T5) | [`core/agent/`](../core/agent/) | [`docs/agents/`](agents/) |
| **Training** | LoRA SFT · DPO (예정) · distillation (예정) | [`core/training/`](../core/training/) | [`docs/training/`](training/) |
| **Serving** | TransformersClient · GGUFClient · AWQ (예정) | [`core/serving/`](../core/serving/) | [`docs/serving/`](serving/) |
| **Evaluation** | Behavioral test · LLM-as-judge · RAG eval | [`core/eval/`](../core/eval/) | [`docs/eval/`](eval/) |

각 영역의 `docs/{area}/README.md` 가 시작점.

## Cross-cutting concerns (영역 가로지름)

이건 어느 한 영역에 속하지 않음 — 영역 간 *조정자* 역할.

| 문서 | 책임 |
|---|---|
| [`ARCHITECTURE.md`](ARCHITECTURE.md) (이 문서) | 영역 지도 · 기여 패턴 |
| [`CAPABILITY_TIERS.md`](CAPABILITY_TIERS.md) | T1-T5 능력 사다리 · 페르소나 ↔ 티어 매핑 |
| [`PERSONA_PACK.md`](PERSONA_PACK.md) | `.adelie` 팩 포맷 (페르소나 영역의 출하 단위) |

## 영역 README 의 표준 템플릿

새 영역을 추가하거나 기존 영역의 README 를 갱신할 때 다음 7 섹션 채움:

```markdown
# {Area Name}

## 책임 — 1줄 요약 + 책임 범위 (이 영역에서 다루지 않는 것 명시)

## 핵심 파일
- `core/{area}/protocols.py` — Protocol 정의 (있으면)
- `core/{area}/{impl}.py` — 구체 구현 (여러 개 가능)

## 현재 상태
- ✅ 구현 완료
- 🔄 진행 중
- ❌ 미구현 (로드맵 참고)

## 사용법 (코드 발췌 1-2)
\`\`\`python
from core.{area} import ...
\`\`\`

## 평가 — 이 영역의 출력을 어떻게 평가하나?
[`docs/eval/`](../eval/) 의 어떤 메서드를 사용? 메트릭 임계?

## 로드맵
- [ ] 다음에 할 일 1
- [ ] 다음에 할 일 2

## 함정
이 영역에서 자주 부딪히는 실수.

## 기여 가이드
PR 어떻게 / 테스트 어디 / 문서 갱신 어디.
```

## 점진적 개선 패턴

새 기능 추가 시:

1. **Protocol 먼저** — `core/{area}/protocols.py` 에 인터페이스 정의 (이미 있으면 갱신)
2. **Stub 구현** — `core/{area}/{impl}_stub.py` 에 최소 동작 (`raise NotImplementedError` 가 아닌 *돌아가는* 가짜)
3. **테스트** — `tests/test_{area}_{feature}.py` — Protocol 적합 + 기본 행동
4. **실구현** — `core/{area}/{impl}.py` 에 진짜 백엔드 (rdflib · transformers 등)
5. **build_app 디스패치** — 의존성 있으면 실구현, 없으면 stub fallback (graceful degrade)
6. **문서** — `docs/{area}/{feature}.md` 또는 README 갱신
7. **로드맵 갱신** — `docs/{area}/README.md` 의 "현재 상태" 업데이트

이 패턴이 [`core/retrieval/graph_retriever.py`](../core/retrieval/graph_retriever.py) (Protocol) → [`core/retrieval/graph_retriever_stub.py`](../core/retrieval/graph_retriever_stub.py) (Stub) → [`core/retrieval/graph_retriever_rdflib.py`](../core/retrieval/graph_retriever_rdflib.py) (실구현) 에 적용됨. T4 KG-RAG 의 진화 사례.

같은 패턴이 [`core/tools/protocols.py`](../core/tools/protocols.py) → [`core/tools/evidence_search.py`](../core/tools/evidence_search.py) (T3 stub) 에도 적용됨. 향후 실 RAG-as-tool 도 같은 골격.

## 의존성 규칙

영역 간 import 방향:

```
[higher level]  agents  →  tools  →  retrieval  →  serving  →  [lower level]
                  ↓        ↓          ↓             ↓
                personas  ←  personas  ←  personas  ←  personas
                  ↑
                evaluation (cross-cuts all)
```

규칙:
- **personas 가 모든 영역을 사용** — 페르소나 = 다른 모든 능력의 *조립*
- **agents → tools → retrieval → serving** 단방향 (역방향 import 금지)
- **evaluation 은 모든 영역에 의존 가능** — 평가 자체가 다른 영역의 출력을 검사하므로
- **Protocol 정의는 항상 가장 의존성 적은 곳** — `core/serving/protocols.py` 가 transformers 임포트 X

## 새 영역 추가 절차

새 능력 (예: "voice synthesis") 이 어느 7 영역에도 안 맞으면:

1. **이 ARCHITECTURE.md 갱신** — 새 행 추가
2. **`core/{new_area}/`** 디렉터리 생성, `protocols.py` 부터 시작
3. **`docs/{new_area}/README.md`** 표준 템플릿으로 작성
4. **`tests/test_{new_area}_*.py`** Protocol 적합 테스트
5. **`build_app` 통합** (필요 시) — `app.state.{new_area}` 에 attach
6. **`/health` 응답** 확장 (해당 능력의 활성 여부)
7. **`docs/CAPABILITY_TIERS.md`** 갱신 (이 능력이 어느 티어에 추가되는가)

## 기여자에게

처음 기여하면:
1. 이 ARCHITECTURE.md 를 읽고 어느 영역에 기여하는지 결정
2. 그 영역의 `README.md` 의 "기여 가이드" 섹션 따름
3. PR 에 어느 영역인지 명시 (`feat(personas):`, `feat(retrieval):` 등 commit prefix)
4. 영역의 로드맵 항목을 닫는 PR 이면 그 항목 체크박스 ✅ 로 갱신
