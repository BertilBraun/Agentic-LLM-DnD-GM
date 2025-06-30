# Whisper Speech-to-Text – Quick Reference

OpenAI Whisper is a general-purpose automatic speech recognition (ASR) model trained on 680k hours of multilingual data.

This cheat-sheet focuses on the Python package (`pip install openai-whisper`).

## Installation

```bash
pip install --upgrade openai-whisper torch soundfile
```

GPU (CUDA) acceleration requires `torch` built with CUDA (~4× faster).

## Basic Transcription

```python
import whisper
model = whisper.load_model("base")
result = model.transcribe("audio.wav")
print(result["text"])
```

`transcribe()` autodetects language and returns a dict with keys:

* `text` – final transcription.
* `segments` – list of time-stamped segments.
* `language` – ISO-639-1 code.

## Real-time Streaming (chunked)

1. Capture mic at 16 kHz mono.
2. Collect ~5 s chunks.
3. Call `whisper.pad_or_trim()` → `whisper.log_mel_spectrogram()`.
4. Run `whisper.decode(model, mel, opts)` where `opts = whisper.DecodingOptions(...)`.

Example is implemented in `src/audio/stt.py`.

## Models & Performance

| Model  | Size   | VRAM fp16 | EN WER | Notes                        |
| ------ | ------ | --------- | ------ | ---------------------------- |
| tiny   | 39 M   | 1 GB      | 6.7 %  | Fastest, good for prototypes |
| base   | 74 M   | 1 GB      | 6.0 %  | Default in repo              |
| small  | 244 M  | 2 GB      | 5.8 %  | Better accuracy              |
| medium | 769 M  | 5 GB      | 5.2 %  | Near-state-of-the-art        |
| large  | 1550 M | 10 GB     | 4.7 %  | Best accuracy, slowest       |

`small` is a good balance for real-time use on consumer GPUs.

## Decoding Options

```python
opts = whisper.DecodingOptions(
    language="en",
    without_timestamps=True,
    fp16=torch.cuda.is_available(),
    temperature=0.0,
)
```

• `temperature` controls randomness (0 = greedy).  
• `without_timestamps=True` omits per-word time codes.

## Tips

* Resample audio →16 kHz mono before feeding to Whisper.
* For noisy input try `temperature=(0, 0.2, 0.4, 0.6)` – Whisper will pick best.
* Batch multiple chunks for GPU utilisation: `whisper.DecodingTask(model)`. (Advanced.)

## Links

* GitHub: <https://github.com/openai/whisper>  
* Paper: <https://arxiv.org/abs/2212.04356> 