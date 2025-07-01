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

from pathlib import Path
import threading
from typing import Optional
import wave

import numpy as np
import sounddevice as sd
import whisper


class WhisperSTT:
    """Real-time speech-to-text using OpenAI Whisper.

    Example
    -------
    >>> stt = WhisperSTT(model_name='base')
    >>> stt.start()
    >>> sleep(5)
    >>> text = stt.close()
    """

    CHANNELS = 1
    SAMPLE_RATE = 44100

    def __init__(
        self,
        model_name: str = 'base',
        *,
        device: Optional[str] = None,
        language: str = 'en',
        initial_prompt: str = '',
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
        self.language = language
        self.initial_prompt = initial_prompt

        self._model = whisper.load_model(model_name, device=device)

        # Runtime state
        self._buffer: list[np.ndarray] = []
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

        self._buffer.clear()
        self._stop_event.clear()

        self._record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._record_thread.start()

    def close(self) -> str:
        """Stop audio capture and wait for background threads to finish."""
        self._stop_event.set()
        if self._record_thread and self._record_thread.is_alive():
            self._record_thread.join()
            self._record_thread = None

        if not self._buffer:
            return ''

        audio = np.concatenate(self._buffer, axis=0)
        FILENAME = 'audio.wav'
        with wave.open(FILENAME, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.SAMPLE_RATE)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())

        abs_filename = Path(FILENAME).absolute()
        result = self._model.transcribe(
            str(abs_filename), language=self.language, initial_prompt=self.initial_prompt, fp16=False
        )
        Path(abs_filename).unlink()

        return result['text']  # type: ignore

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _record_loop(self) -> None:
        """Continuously read microphone frames into a queue."""
        with sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            dtype='float32',
            callback=self._audio_callback,
        ):
            while not self._stop_event.is_set():
                sd.sleep(50)

    def _audio_callback(self, indata, frames, time_info, status):  # noqa: D401, N802  (sounddevice signature)
        if status:
            print(status, flush=True)
        # indata is shape (frames, channels) – ensure mono float32 numpy
        self._buffer.append(indata.copy())


def stt(model: WhisperSTT) -> str:
    """
    Transcribe the audio from the microphone.
    """
    input('Press Enter to start recording...')
    model.start()
    input('Press Enter to stop recording...')
    text = model.close()
    print(f'🗣️  Player: {text}')
    return text


if __name__ == '__main__':
    """Quick manual test: stream transcripts until keyboard interrupt."""
    from time import sleep

    model = WhisperSTT(model_name='base')
    print('Starting microphone transcription for 5 seconds...')
    model.start()
    sleep(5)
    print(model.close())
