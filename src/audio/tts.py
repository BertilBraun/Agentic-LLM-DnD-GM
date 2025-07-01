"""Text-to-Speech abstraction layer.

Currently supports:
    • "coqui" – via the `TTS` Python package (https://github.com/coqui-ai/TTS)
    • "openai" – via the `openai` Python package (https://github.com/openai/openai)

The design follows a simple adapter pattern so that additional engines (e.g.
Piper, Azure, ElevenLabs) can be plugged in later with minimal effort.
"""

from __future__ import annotations

import hashlib
import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Generator, Iterable, Type
import os
from openai import OpenAI

import numpy as np
import soundfile as sf


class BaseTTS(ABC):
    """Abstract base class for all TTS adapters."""

    @abstractmethod
    def speak(self, text: str) -> Generator[bytes, None, None]:
        """Convert *text* to speech.

        Returns
        -------
        bytes
            WAV-encoded PCM data.
        """

    def play(self, text: str) -> None:
        """Play the audio."""
        play_tts(self.speak(text))


# ---------------------------------------------------------------------------
# Coqui / TTS implementation
# ---------------------------------------------------------------------------
class CoquiTTSAdapter(BaseTTS):
    """Wrapper around the Coqui TTS library (`TTS` package)."""

    def __init__(self, model_name: str | None = None, *, gpu: bool = False) -> None:
        from TTS.api import TTS as _CoquiTTS  # type: ignore

        self._model = _CoquiTTS(model_name=model_name, progress_bar=False, gpu=gpu)

    def speak(self, text: str) -> Generator[bytes, None, None]:  # noqa: D401
        # Generate raw audio and sample rate
        audio, sample_rate = self._model.tts(text)
        # Encode to WAV in-memory
        with io.BytesIO() as buf:
            sf.write(buf, audio.astype(np.float32), sample_rate, format='WAV')
            yield buf.getvalue()


class OpenAITTSAdapter(BaseTTS):
    """Wrapper around the OpenAI TTS library (`TTS` package)."""

    def __init__(self, api_key: str, voice_model: str, voice_id: str, instructions: str) -> None:
        self._client = OpenAI(api_key=api_key)
        self._voice_model = voice_model
        self._voice_id = voice_id
        self._instructions = instructions

    def speak(self, text: str) -> Generator[bytes, None, None]:  # noqa: D401
        id_hash = hashlib.sha256(
            text.encode() + self._instructions.encode() + self._voice_id.encode() + self._voice_model.encode()
        ).hexdigest()
        output_path = Path(f'cache/tts/{id_hash}.wav')

        if output_path.exists():
            with open(output_path, 'rb') as f:
                yield f.read()

        else:
            with self._client.audio.speech.with_streaming_response.create(
                model=self._voice_model,
                voice=self._voice_id,
                input=text,
                instructions=self._instructions,
                response_format='pcm',
                speed=1.2,
            ) as response:
                buffer = io.BytesIO()

                for chunk in response.iter_bytes(1024):
                    yield chunk
                    buffer.write(chunk)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(buffer.getvalue())


def play_tts(audio: Iterable[bytes]) -> None:
    import pyaudio

    p = pyaudio.PyAudio()
    stream = p.open(format=8, channels=1, rate=24_000, output=True)
    for chunk in audio:
        stream.write(chunk)
    stream.stop_stream()
    stream.close()
    p.terminate()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_ADAPTERS: Dict[str, Type[BaseTTS]] = {
    'coqui': CoquiTTSAdapter,
    'openai': OpenAITTSAdapter,
}

# Default engine can be overridden via env or at runtime
_default_engine: str = os.getenv('TTS_ENGINE', 'coqui').lower()


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


if __name__ == '__main__':
    tts = get_tts(engine='openai', api_key=os.getenv('OPENAI_API_KEY'), voice_model='tts-1', voice_id='alloy')
    wav_bytes_iterable = tts.speak('Hello, world!')
    play_tts(wav_bytes_iterable)
