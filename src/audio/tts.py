"""Text-to-Speech abstraction layer.

Currently supports:
    • "coqui" – via the `TTS` Python package (https://github.com/coqui-ai/TTS)

The design follows a simple adapter pattern so that additional engines (e.g.
Piper, Azure, ElevenLabs) can be plugged in later with minimal effort.
"""
from __future__ import annotations

import io
import importlib
from abc import ABC, abstractmethod
from typing import Dict, Type
import os

import numpy as np  # type: ignore
import soundfile as sf  # type: ignore


class BaseTTS(ABC):
    """Abstract base class for all TTS adapters."""

    @abstractmethod
    def speak(self, text: str) -> bytes:  # noqa: D401
        """Convert *text* to speech.

        Returns
        -------
        bytes
            WAV-encoded PCM data.
        """

    # You could also add async counterpart if needed.


# ---------------------------------------------------------------------------
# Coqui / TTS implementation
# ---------------------------------------------------------------------------
class CoquiTTSAdapter(BaseTTS):
    """Wrapper around the Coqui TTS library (`TTS` package)."""

    def __init__(self, model_name: str | None = None, *, gpu: bool = False) -> None:
        try:
            from TTS.api import TTS as _CoquiTTS  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover – handled at runtime
            raise RuntimeError(
                "Coqui TTS package not installed. Add `TTS` to requirements.txt"
            ) from exc

        self._model = _CoquiTTS(model_name=model_name, progress_bar=False, gpu=gpu)

    def speak(self, text: str) -> bytes:  # noqa: D401
        # Generate raw audio and sample rate
        audio, sample_rate = self._model.tts(text)
        # Encode to WAV in-memory
        with io.BytesIO() as buf:
            sf.write(buf, audio.astype(np.float32), sample_rate, format="WAV")
            return buf.getvalue()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_ADAPTERS: Dict[str, Type[BaseTTS]] = {
    "coqui": CoquiTTSAdapter,
    # "piper": PiperTTSAdapter,  # To be implemented
}

# Default engine can be overridden via env or at runtime
_default_engine: str = os.getenv("TTS_ENGINE", "coqui").lower()


def set_default_engine(engine: str) -> None:
    """Set global default TTS *engine* for subsequent :func:`get_tts` calls."""
    engine = engine.lower()
    if engine not in _ADAPTERS:
        raise ValueError(f"Unknown TTS engine '{engine}'. Available: {list(_ADAPTERS)}")
    global _default_engine
    _default_engine = engine


def get_tts(engine: str | None = None, **kwargs) -> BaseTTS:
    """Return a TTS adapter instance.

    Parameters
    ----------
    engine
        Engine name to instantiate. If *None*, the module-level default is used.
    """
    if engine is None:
        engine = _default_engine
    engine = engine.lower()
    try:
        adapter_cls = _ADAPTERS[engine]
    except KeyError as exc:  # pragma: no cover – runtime selection
        raise ValueError(f"Unknown TTS engine '{engine}'. Available: {list(_ADAPTERS)}") from exc
    return adapter_cls(**kwargs)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# CLI quick-start
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import os
    import uuid

    import sounddevice as sd  # type: ignore

    parser = argparse.ArgumentParser(description="Simple TTS CLI test")
    parser.add_argument("text", help="Text to read aloud")
    parser.add_argument("--engine", default="coqui", help="TTS engine name")
    args = parser.parse_args()

    tts = get_tts(args.engine)
    wav_bytes = tts.speak(args.text)

    # Save to temp file then play via sounddevice
    tmp_path = f"/tmp/{uuid.uuid4().hex}.wav" if os.name != "nt" else f"{uuid.uuid4().hex}.wav"
    with open(tmp_path, "wb") as f:
        f.write(wav_bytes)

    print(f"Playing generated speech ({len(wav_bytes)/1024:.1f} KB)...")
    data, sr = sf.read(tmp_path, dtype="float32")
    sd.play(data, sr)
    sd.wait()

    os.remove(tmp_path) 