"""Speech-to-Text wrapper for real-time microphone transcription using OpenAI Whisper.

This module captures audio from the system microphone, feeds it into a loaded
Whisper model, and yields transcript strings as soon as they are ready.

Dependencies (see requirements.txt):
    • torch
    • openai-whisper
    • sounddevice
    • soundfile (required by sounddevice)
"""
from __future__ import annotations

import queue
import threading
import time
from typing import Generator, Optional

import numpy as np  # type: ignore
import sounddevice as sd  # type: ignore
import whisper  # type: ignore


class WhisperSTT:
    """Real-time speech-to-text using OpenAI Whisper.

    Example
    -------
    >>> stt = WhisperSTT(model_name="base")
    >>> stt.start()
    >>> for text in stt.stream():
    ...     print(text)
    ...     if "stop" in text.lower():
    ...         break
    >>> stt.close()
    """

    def __init__(
        self,
        model_name: str = "base",
        *,
        device: Optional[str] = None,
        sample_rate: int = 16_000,
        chunk_seconds: int = 5,
        num_mel_samples: int = 30,
    ) -> None:
        """Create a new :class:`WhisperSTT` instance.

        Parameters
        ----------
        model_name
            Whisper checkpoint to load (``tiny``, ``base``, ``small``, ``medium``, ``large``...).
        device
            Torch device string (e.g. ``"cuda"``). If *None*, Whisper auto-selects.
        sample_rate
            Microphone sampling rate in Hz. Whisper models expect 16 kHz.
        chunk_seconds
            Target length (in seconds) of audio chunks that are fed to Whisper.
        num_mel_samples
            Number of mel frames Whisper uses for padding/trimming. Leave default unless you
            know what you are doing.
        """
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.num_mel_samples = num_mel_samples

        self._model = whisper.load_model(model_name, device=device)

        # Runtime state
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._record_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def start(self) -> None:
        """Begin capturing microphone audio in a background thread."""
        if self._record_thread and self._record_thread.is_alive():
            # Already started
            return

        self._stop_event.clear()
        self._record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._record_thread.start()

    def stream(self) -> Generator[str, None, None]:
        """Yield transcribed text as soon as each audio chunk is processed."""
        buffer: list[np.ndarray] = []
        last_flush = time.time()

        while not self._stop_event.is_set():
            try:
                data = self._audio_queue.get(timeout=0.1)
            except queue.Empty:
                # Periodically flush even if no new data – so we don't block on the last chunk.
                if buffer and (time.time() - last_flush) >= self.chunk_seconds:
                    text = self._transcribe(np.concatenate(buffer, axis=0))
                    if text:
                        yield text
                    buffer.clear()
                    last_flush = time.time()
                continue

            buffer.append(data)
            if (time.time() - last_flush) >= self.chunk_seconds:
                # Enough audio collected – send to Whisper
                audio = np.concatenate(buffer, axis=0)
                buffer.clear()
                text = self._transcribe(audio)
                last_flush = time.time()
                if text:
                    yield text

    def close(self) -> None:
        """Stop audio capture and wait for background threads to finish."""
        self._stop_event.set()
        if self._record_thread and self._record_thread.is_alive():
            self._record_thread.join()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _record_loop(self) -> None:
        """Continuously read microphone frames into a queue."""
        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        ):
            while not self._stop_event.is_set():
                sd.sleep(50)

    def _audio_callback(self, indata, frames, time_info, status):  # noqa: D401, N802  (sounddevice signature)
        if status:
            print(status, flush=True)
        # indata is shape (frames, channels) – ensure mono float32 numpy
        self._audio_queue.put(indata.copy())

    def _transcribe(self, audio: np.ndarray) -> str:
        """Run Whisper on *audio* and return the transcription."""
        if audio.size == 0:
            return ""

        # Whisper expects (n,) float32 PCM at sample_rate Hz
        # Pad/trim to target length (num_mel_samples ≈ 30 for ~18 s @16kHz)
        audio = whisper.pad_or_trim(audio.flatten(), length=self.sample_rate * self.chunk_seconds)
        mel = whisper.log_mel_spectrogram(audio, padding=0).to(self._model.device)

        options = whisper.DecodingOptions(
            language="en",
            without_timestamps=True,
            fp16=False,
        )
        result = whisper.decode(self._model, mel, options)
        return result.text.strip()


if __name__ == "__main__":
    """Quick manual test: stream transcripts until keyboard interrupt."""

    stt = WhisperSTT(model_name="base")
    try:
        print("Starting microphone transcription... Press Ctrl+C to stop.")
        stt.start()
        for transcript in stt.stream():
            if transcript:
                print("> ", transcript)
    except KeyboardInterrupt:
        pass
    finally:
        stt.close()
        print("STT stopped.") 