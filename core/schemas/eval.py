from pydantic import BaseModel, ConfigDict, Field


class MetricScore(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    score: float = Field(ge=0.0, le=1.0)
    details: dict = Field(default_factory=dict)


class EvalResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    faithfulness: float | None = None
    answer_relevance: float | None = None
    citation_coverage: float | None = None
    retrieval_recall_at_k: float | None = None
    metrics: list[MetricScore] = Field(default_factory=list)
    notes: str | None = None
    evaluator_model_id: str | None = None
