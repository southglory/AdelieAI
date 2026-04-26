import asyncio
import json
import threading
import time
from pathlib import Path
from typing import AsyncIterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from core.serving.protocols import GenerationParams, GenerationResult, StreamEvent


class TransformersClient:
    def __init__(
        self,
        model_path: str | Path,
        *,
        device_map: str = "auto",
        dtype: torch.dtype | str = torch.bfloat16,
        lora_path: str | Path | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        manifest_path = self.model_path / "MANIFEST.json"
        if manifest_path.exists():
            self._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.model_id = self._manifest.get("model_id", str(self.model_path))
        else:
            self._manifest = None
            self.model_id = str(self.model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=dtype,
            device_map=device_map,
        )

        self.lora_path = Path(lora_path) if lora_path else None
        if self.lora_path is not None and self.lora_path.exists():
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, str(self.lora_path))
            adapter_manifest = self.lora_path / "MANIFEST.json"
            if adapter_manifest.exists():
                adapter_data = json.loads(
                    adapter_manifest.read_text(encoding="utf-8")
                )
                adapter_id = adapter_data.get("model_id", self.lora_path.name)
                self.model_id = f"{self.model_id}+{adapter_id}"

        self.model.eval()

    def _build_prompt(self, prompt: str, system: str | None) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _gen_kwargs(self, params: GenerationParams) -> dict:
        do_sample = params.temperature > 0.0
        kwargs = {
            "max_new_tokens": params.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs["temperature"] = params.temperature
            kwargs["top_p"] = params.top_p
            kwargs["top_k"] = params.top_k
        return kwargs

    def _generate_sync(
        self, prompt: str, params: GenerationParams
    ) -> GenerationResult:
        templated = self._build_prompt(prompt, params.system)
        inputs = self.tokenizer(templated, return_tensors="pt").to(self.model.device)
        tokens_in = int(inputs.input_ids.shape[1])

        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**inputs, **self._gen_kwargs(params))
        latency_ms = int((time.perf_counter() - t0) * 1000)

        gen_tokens = out[0][inputs.input_ids.shape[1] :]
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
        return GenerationResult(
            text=text,
            tokens_in=tokens_in,
            tokens_out=int(gen_tokens.shape[0]),
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
        templated = self._build_prompt(prompt, params.system)
        inputs = self.tokenizer(templated, return_tensors="pt").to(self.model.device)
        tokens_in = int(inputs.input_ids.shape[1])

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = {**self._gen_kwargs(params), "streamer": streamer, **inputs}

        def _generate_in_thread() -> None:
            with torch.no_grad():
                self.model.generate(**gen_kwargs)

        t0 = time.perf_counter()
        thread = threading.Thread(target=_generate_in_thread, daemon=True)
        thread.start()

        chunks: list[str] = []
        loop = asyncio.get_event_loop()
        try:
            while True:
                chunk = await loop.run_in_executor(
                    None, lambda: next(streamer, _SENTINEL)
                )
                if chunk is _SENTINEL:
                    break
                chunks.append(chunk)
                yield StreamEvent(type="chunk", text=chunk)
        finally:
            thread.join(timeout=5.0)

        text = "".join(chunks)
        tokens_out = len(self.tokenizer.encode(text, add_special_tokens=False))
        yield StreamEvent(
            type="done",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=int((time.perf_counter() - t0) * 1000),
        )


_SENTINEL = object()
