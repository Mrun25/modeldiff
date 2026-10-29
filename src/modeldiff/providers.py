"""Inference providers.

A provider turns a test case into a model output. Two are included:

* ``replay`` reads outputs recorded earlier from a JSON file. It needs no
  model and is what the examples and test suite use.
* ``ollama`` calls a local Ollama server over HTTP using only the standard
  library. It is never used unless a config selects it.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from modeldiff.cases import TestCase, read_document
from modeldiff.config import RunConfig
from modeldiff.errors import ProviderError, ValidationError


@dataclass
class ProviderResponse:
    output: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # Latency reported by the source instead of measured around the call,
    # e.g. a latency stored alongside a recorded output.
    recorded_latency_ms: float | None = None


class Provider(Protocol):
    name: str

    def generate(self, case: TestCase) -> ProviderResponse: ...

    def describe(self) -> dict[str, Any]:
        """Runtime and model metadata recorded once per run."""
        ...


class ReplayProvider:
    """Serve outputs recorded in a JSON file.

    File format::

        {"outputs": {
            "case-id": "plain output text",
            "other-id": {"output": "text", "latency_ms": 812.5, "metadata": {...}}
        }}
    """

    name = "replay"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        data = read_document(self.path)
        if not isinstance(data, dict) or not isinstance(data.get("outputs"), dict):
            raise ValidationError(str(self.path), ["expected a mapping with an 'outputs' mapping"])
        problems = []
        for case_id, entry in data["outputs"].items():
            if isinstance(entry, dict):
                if not isinstance(entry.get("output"), str):
                    problems.append(f"{case_id}: 'output' must be a string")
                lat = entry.get("latency_ms")
                if lat is not None and (not isinstance(lat, int | float) or lat < 0):
                    problems.append(f"{case_id}: 'latency_ms' must be a non-negative number")
            elif not isinstance(entry, str):
                problems.append(f"{case_id}: entry must be a string or a mapping with 'output'")
        if problems:
            raise ValidationError(str(self.path), problems)
        self._outputs: dict[str, Any] = data["outputs"]
        self._meta: dict[str, Any] = (
            data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
        )

    @property
    def recorded_case_ids(self) -> frozenset[str]:
        """Case IDs available in the recorded-output file."""
        return frozenset(self._outputs)

    def generate(self, case: TestCase) -> ProviderResponse:
        entry = self._outputs.get(case.id)
        if entry is None:
            raise ProviderError(f"no recorded output for case {case.id!r} in {self.path}")
        if isinstance(entry, str):
            return ProviderResponse(entry)
        return ProviderResponse(
            entry["output"], dict(entry.get("metadata", {})), entry.get("latency_ms")
        )

    def describe(self) -> dict[str, Any]:
        return {"provider": "replay", "outputs_file": self.path.name, **self._meta}


class OllamaProvider:
    """Generate outputs with a local Ollama server (``/api/generate``)."""

    name = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        options: dict[str, Any] | None = None,
        format: Any = None,
        system: str | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.options = options or {}
        self.format = format
        self.system = system
        self.timeout_s = timeout_s

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise ProviderError(f"Ollama returned HTTP {exc.code}: {detail}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            reason = getattr(exc, "reason", exc)
            raise ProviderError(
                f"could not reach Ollama at {self.base_url} ({reason}); is `ollama serve` running?"
            ) from None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            raise ProviderError("Ollama returned a response that is not JSON") from None

    def generate(self, case: TestCase) -> ProviderResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": case.input,
            "stream": False,
            # Case-level generation settings override the config's options.
            "options": {**self.options, **case.generation},
        }
        if self.format is not None:
            payload["format"] = self.format
        if self.system is not None:
            payload["system"] = self.system
        data = self._post("/api/generate", payload)
        if "error" in data:
            raise ProviderError(f"Ollama error: {data['error']}")
        if not isinstance(data.get("response"), str):
            raise ProviderError("Ollama response has no 'response' text")
        keys = (
            "model",
            "total_duration",
            "load_duration",
            "prompt_eval_count",
            "eval_count",
            "eval_duration",
        )
        return ProviderResponse(data["response"], {k: data[k] for k in keys if k in data})

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "provider": "ollama",
            "model": self.model,
            "base_url": self.base_url,
        }
        try:
            version = self._get_version()
        except ProviderError:
            version = None
        if version:
            info["ollama_version"] = version
        return info

    def _get_version(self) -> str | None:
        try:
            with urllib.request.urlopen(self.base_url + "/api/version", timeout=5) as response:
                return json.loads(response.read()).get("version")
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError):
            raise ProviderError("version lookup failed") from None


def build_provider(config: RunConfig) -> Provider:
    s = config.settings
    if config.provider == "replay":
        return ReplayProvider(config.base_dir / s["outputs"])
    if config.provider == "ollama":
        return OllamaProvider(
            model=s["model"],
            base_url=s.get("base_url", "http://localhost:11434"),
            options=s.get("options"),
            format=s.get("format"),
            system=s.get("system"),
            timeout_s=s.get("timeout_s", 120.0),
        )
    raise ValidationError("config", [f"unknown provider {config.provider!r}"])
