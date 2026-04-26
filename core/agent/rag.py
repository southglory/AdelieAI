from core.schemas.retrieval import RetrievedContext


def format_context(query: str, ctx: RetrievedContext) -> str:
    """Wrap retrieved chunks into a prompt the LLM can lean on. Uses
    the standard 'context block + question' shape from retrieval-
    augmented QA literature; numbered citations let the model refer
    back ([1], [2], …).
    """
    if not ctx.results:
        return query
    blocks = []
    for i, r in enumerate(ctx.results, start=1):
        title = r.chunk.metadata.get("doc_title")
        header = f"[{i}]" + (f" ({title})" if title else "")
        blocks.append(f"{header}\n{r.chunk.text}")
    body = "\n\n".join(blocks)
    return (
        "다음 컨텍스트를 활용해 마지막 질문에 답하세요. "
        "컨텍스트가 답을 포함하지 않으면 모른다고 정직하게 말하세요. "
        "근거 인용은 [숫자] 형식.\n\n"
        f"--- Context ---\n{body}\n--- End Context ---\n\n"
        f"Question: {query}"
    )


def retrieval_event_payload(ctx: RetrievedContext) -> dict:
    return {
        "method": ctx.method,
        "query": ctx.query,
        "results": [
            {
                "score": r.score,
                "chunk_id": r.chunk.id,
                "doc_id": r.chunk.doc_id,
                "position": r.chunk.position,
                "doc_title": r.chunk.metadata.get("doc_title"),
                "preview": r.chunk.text[:200],
            }
            for r in ctx.results
        ],
    }
