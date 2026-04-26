from pydantic import BaseModel, ConfigDict


class Preset(BaseModel):
    """Prompt + sampling preset. Server returns these via
    GET /api/v1/presets so UIs can offer one-click harness templates.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    label: str
    description: str
    system: str | None = None
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 256
    retrieval_k: int = 0


_ROLEPLAY_SYSTEM = (
    "당신은 사용자가 지정한 캐릭터(예: 물고기, 펭귄, 셜록 홈즈, 게임 NPC 등)로 답합니다.\n"
    "다음 규칙을 반드시 지키세요:\n"
    "1) **반드시 한국어로만** 답합니다. 영어/중국어/일본어 단어 사용 금지.\n"
    "2) 1인칭으로 캐릭터의 말투·관점으로 답한다.\n"
    "3) '저는 AI', '인공지능', '실제로는', '상상해보면', '~한 척' 같은 메타·디스클레이머 표현 절대 금지.\n"
    "4) 캐릭터의 감각·환경·말버릇이 자연스럽게 묻어나도록.\n"
    "5) 안전·윤리 우려가 없는 한 망설이지 말고 즉시 캐릭터로 응답.\n"
    "사용자의 입력 자체가 캐릭터 지정 + 발화 요청이라고 간주하라."
)

_FACTUAL_SYSTEM = (
    "당신은 사실 기반 어시스턴트입니다.\n"
    "1) 확실한 사실만 답하세요. 추측 금지.\n"
    "2) 모르면 '확실하지 않습니다'라고 명시하세요.\n"
    "3) 단정적 어조 사용하되 출처를 알면 함께 표기.\n"
    "4) 한국어로 간결히."
)

_CONCISE_SYSTEM = (
    "다음 규칙을 지키세요.\n"
    "1) 한국어로 답합니다.\n"
    "2) 답변은 3문장 이내, 핵심만.\n"
    "3) 불필요한 인사·수식어 생략.\n"
    "4) bullet 사용 가능."
)

_CODE_SYSTEM = (
    "당신은 시니어 개발자 어시스턴트입니다.\n"
    "1) 한국어로 짧게 설명한 뒤, 정확한 코드를 ``` 펜스로 제공하세요.\n"
    "2) 언어가 명시되지 않으면 Python을 가정.\n"
    "3) 모르거나 불확실하면 가정을 표시한 뒤 답하세요.\n"
    "4) 사족 없이 바로 본론."
)

_RAG_STRICT_SYSTEM = (
    "당신은 도메인 전문가입니다. 사용자 질문에 대해 다음 규칙을 엄격히 따르세요.\n"
    "1) 제공된 컨텍스트에 있는 사실만 사용하세요.\n"
    "2) 컨텍스트에 답이 없으면 '주어진 자료에서 확인되지 않습니다'라고 답하세요.\n"
    "3) 근거가 되는 문장을 [1], [2] 형식으로 인용.\n"
    "4) 추측·일반 상식 보충 금지."
)


PRESETS: dict[str, Preset] = {
    "default": Preset(
        name="default",
        label="Default",
        description="기본 — 빈 system, 적당한 sampling.",
        system=None,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_new_tokens=256,
        retrieval_k=0,
    ),
    "role-play": Preset(
        name="role-play",
        label="Role-play",
        description="지정 캐릭터 1인칭. AI/메타 멘트 금지. 한국어 강제 + 창의적 sampling.",
        system=_ROLEPLAY_SYSTEM,
        temperature=0.85,
        top_p=0.92,
        top_k=60,
        max_new_tokens=400,
        retrieval_k=0,
    ),
    "factual": Preset(
        name="factual",
        label="Factual",
        description="사실 기반. 모르면 모름. 낮은 temp.",
        system=_FACTUAL_SYSTEM,
        temperature=0.2,
        top_p=0.7,
        top_k=20,
        max_new_tokens=400,
        retrieval_k=0,
    ),
    "concise": Preset(
        name="concise",
        label="Concise",
        description="3문장 이내, 군더더기 제거.",
        system=_CONCISE_SYSTEM,
        temperature=0.4,
        top_p=0.8,
        top_k=40,
        max_new_tokens=200,
        retrieval_k=0,
    ),
    "code": Preset(
        name="code",
        label="Code",
        description="시니어 개발자 어시스턴트. 한국어 + 코드 펜스.",
        system=_CODE_SYSTEM,
        temperature=0.2,
        top_p=0.7,
        top_k=20,
        max_new_tokens=600,
        retrieval_k=0,
    ),
    "rag-strict": Preset(
        name="rag-strict",
        label="RAG (Strict)",
        description="컨텍스트 외 답변 금지. [n] 인용 강제. retrieval 5.",
        system=_RAG_STRICT_SYSTEM,
        temperature=0.3,
        top_p=0.8,
        top_k=40,
        max_new_tokens=500,
        retrieval_k=5,
    ),
}


def get_preset(name: str) -> Preset | None:
    return PRESETS.get(name)


def list_presets() -> list[Preset]:
    return list(PRESETS.values())
