"""LLMClient implementation backed by llama-cpp-python (GGUF).

Loads a single `.gguf` file (typically the q4_k_m quantized variant
produced by `differentia-llm/experiments/06_gguf_export/run.py`).

Why GGUF here:
  * Same chat template as the source HF model (preserved by
    convert_hf_to_gguf.py), so prompt formatting matches the
    TransformersClient path.
  * llama-cpp-python's prebuilt CPU wheel works on Windows where
    autoawq cannot install.
  * Quantized footprint (~4.5 GB for 7B q4_k_m) is the v0.2 "ships
    at deployable size" milestone.

The CUDA wheel is harder to source on Windows; this client defaults
to CPU. To run on GPU, install the CUDA build separately and pass
`n_gpu_layers=-1` via the constructor.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import AsyncIterator

from core.serving.protocols import GenerationParams, GenerationResult, StreamEvent


class GGUFClient:
    def __init__(
        self,
        model_path: str | Path,
        *,
        n_ctx: int = 2048,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file() or self.model_path.suffix != ".gguf":
            raise ValueError(
                f"GGUFClient expects a single .gguf file, got: {self.model_path}"
            )

        # Sibling MANIFEST.json is optional; we derive a friendly id from it
        # when present, otherwise fall back to the file stem.
        manifest_path = self.model_path.parent / "MANIFEST.json"
        if manifest_path.exists():
            self._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.model_id = self._manifest.get("model_id", self.model_path.stem)
        else:
            self._manifest = None
            self.model_id = self.model_path.stem

        from llama_cpp import Llama  # type: ignore[import-not-found]

        self._llm = Llama(
            model_path=str(self.model_path),
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )

    def _build_messages(
        self, prompt: str, system: str | None
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _completion_kwargs(self, params: GenerationParams) -> dict:
        do_sample = params.temperature > 0.0
        kwargs: dict = {"max_tokens": params.max_new_tokens}
        if do_sample:
            kwargs.update(
                temperature=params.temperature,
                top_p=params.top_p,
                top_k=params.top_k,
            )
        else:
            kwargs.update(temperature=0.0)
        return kwargs

    def _generate_sync(
        self, prompt: str, params: GenerationParams
    ) -> GenerationResult:
        t0 = time.perf_counter()
        out = self._llm.create_chat_completion(
            messages=self._build_messages(prompt, params.system),
            **self._completion_kwargs(params),
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)

        text = out["choices"][0]["message"]["content"] or ""
        usage = out.get("usage", {})
        return GenerationResult(
            text=text,
            tokens_in=int(usage.get("prompt_tokens", 0)),
            tokens_out=int(usage.get("completion_tokens", 0)),
            latency_ms=latency_ms,
            model_id=self.model_id,
            params=params,
        )

    async def generate(
        self, prompt: str, params: GenerationParams | None = None
    ) -> GenerationResult:
        params = params or GenerationParams()
        return await asyncio.to_thread(self._generate_sync, prompt, params)

    async def astream(
        self, prompt: str, params: GenerationParams | None = None
    ) -> AsyncIterator[StreamEvent]:
        params = params or GenerationParams()
        loop = asyncio.get_event_loop()
        t0 = time.perf_counter()

        def _start_stream():
            return self._llm.create_chat_completion(
                messages=self._build_messages(prompt, params.system),
                stream=True,
                **self._completion_kwargs(params),
            )

        # llama-cpp-python's stream is a synchronous generator; bridge to async.
        gen = await loop.run_in_executor(None, _start_stream)

        chunks: list[str] = []
        try:
            while True:
                event = await loop.run_in_executor(
                    None, lambda: next(gen, _SENTINEL)
                )
                if event is _SENTINEL:
                    break
                delta = event["choices"][0].get("delta", {})
                piece = delta.get("content")
                if piece:
                    chunks.append(piece)
                    yield StreamEvent(type="chunk", text=piece)
        finally:
            # llama-cpp-python's generator does not need explicit close, but
            # be defensive in case future versions add resource handles.
            close = getattr(gen, "close", None)
            if callable(close):
                close()

        text = "".join(chunks)
        # llama-cpp-python's streamed response does not expose usage in the
        # final delta, so we tokenize after-the-fact for tokens_out and
        # estimate tokens_in from the rendered prompt.
        tokens_out = len(self._llm.tokenize(text.encode("utf-8"), add_bos=False))
        # tokens_in: re-tokenize the user prompt + system; this slightly
        # under-counts vs. what the chat template adds, but stays cheap.
        prompt_text = (params.system or "") + "\n" + prompt
        tokens_in = len(
            self._llm.tokenize(prompt_text.encode("utf-8"), add_bos=False)
        )
        yield StreamEvent(
            type="done",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )


_SENTINEL = object()
