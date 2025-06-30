# Text-to-Speech Engines – Piper & Coqui

This document collects installation and usage notes for two open-source TTS engines currently under evaluation.

---

## 1. Piper (Rhasspy)

Piper is a fast neural TTS engine written in Rust, created by the Rhasspy voice-assistant community.

### Key Points

* **Models**: Onnx-runtime voices (~50-100 MB) for many languages; speakers encoded in model name.
* **Performance**: ~4× real-time on Raspberry Pi 4; >20× RT on desktop CPU.
* **Quality**: Comparable to Tacotron2 + HiFi-GAN; depends on voice dataset quality.
* **Licensing**: Apache-2.0 engine; models carry dataset-specific licenses (mostly CC-BY 4.0).

### Install

```bash
pip install piper-tts  # thin Python wrapper
# or native binary
cargo install --git https://github.com/rhasspy/piper piper
```

Download a voice, e.g. **en_US-amy-high** (96 kHz):

```bash
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/en_US-amy-high.onnx
```

### Basic Usage (Python)

```python
from piper.tts import PiperVoice

voice = PiperVoice(load_path="en_US-amy-high.onnx", config_path="en_US-amy-high.onnx.json")
audio = voice.tts("Welcome to the dungeon!")  # returns 16-bit PCM @ 16 kHz
with open("out.wav", "wb") as f:
    f.write(audio)
```

### Pros / Cons

| Pros                      | Cons                        |
| ------------------------- | --------------------------- |
| Tiny CPU-friendly         | Limited prosody control     |
| Voices for many languages | Separate download per voice |

---

## 2. Coqui TTS

Coqui-AI's **TTS** library is a full-featured research/production toolkit supporting multiple TTS architectures (Tacotron2, Glow-TTS, VITS, etc.).

### Key Points

* **Models**: Pre-trained checkpoints on Hugging Face; community voices available.
* **Performance**: GPU recommended for <1× RT; CPU ~0.3× RT on «base» models.
* **Quality**: VITS models reach near-studio quality; multispeaker, emotion control.
* **Licensing**: MIT library; models vary (check README).

### Install

```bash
pip install TTS==0.15.3
# Optional: GPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Basic Usage

```python
from TTS.api import TTS

tts = TTS(model_name="tts_models/en/vctk/vits")
wav = tts.tts("Roll initiative!")  # returns np.ndarray(float32) + sr attribute
TTS.utils.io.save_wav(wav, "out.wav", tts.synthesizer.output_sample_rate)
```

### Pros / Cons

| Pros                              | Cons                 |
| --------------------------------- | -------------------- |
| High-quality neural TTS           | Heavier dependencies |
| Supports emotions, speaker mixing | GPU recommended      |

---

## Selection Rationale

We ship **Coqui** as the default engine because it offers higher quality and direct Python API (used in `src/audio/tts.py`).  
Piper can be integrated later as a fallback CPU-only option.

---

## Links

* Piper repo: <https://github.com/rhasspy/piper>  
* Coqui-AI TTS repo: <https://github.com/coqui-ai/TTS>  
* Hugging Face model hub: <https://huggingface.co/models?library=tts> 