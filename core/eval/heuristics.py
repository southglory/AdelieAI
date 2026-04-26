import re

_CITATION_RE = re.compile(r"\[(\d+)\]")


def citation_coverage(answer: str, retrieved_count: int) -> tuple[float, dict]:
    """How well the answer cites the retrieved contexts.

    Returns (score, details).
    score = (valid in-range citations) / max(retrieved_count, 1).
    Hits the 1.0 ceiling when every retrieved chunk is referenced.
    Out-of-range citations (e.g. [99] when only 3 chunks retrieved)
    are penalised — they suggest the model invented references.
    """
    if retrieved_count <= 0:
        cited = sorted({int(m) for m in _CITATION_RE.findall(answer)})
        return (0.0, {"cited": cited, "retrieved": 0, "out_of_range": cited})

    cited = sorted({int(m) for m in _CITATION_RE.findall(answer)})
    in_range = [c for c in cited if 1 <= c <= retrieved_count]
    out_of_range = [c for c in cited if c not in in_range]
    score = min(1.0, len(in_range) / retrieved_count)
    return (
        score,
        {
            "cited": cited,
            "retrieved": retrieved_count,
            "in_range": in_range,
            "out_of_range": out_of_range,
        },
    )


def retrieval_recall_at_k(
    retrieved_chunk_ids: list[str], ground_truth_chunk_ids: list[str]
) -> tuple[float, dict]:
    """How many of the labelled-relevant chunks made it into the top-k.

    Useful only when ground truth is available; without labels callers
    skip this metric and rely on the LLM-as-judge ones below.
    """
    if not ground_truth_chunk_ids:
        return (0.0, {"reason": "no ground truth"})
    hits = [cid for cid in ground_truth_chunk_ids if cid in retrieved_chunk_ids]
    score = len(hits) / len(ground_truth_chunk_ids)
    return (score, {"hits": len(hits), "ground_truth_size": len(ground_truth_chunk_ids)})
