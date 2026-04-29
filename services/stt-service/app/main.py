import asyncio
import base64
import os
import tempfile

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title='stt-service')

STT_PROVIDER = os.environ.get('STT_PROVIDER', 'whisper_local')
WHISPER_MODEL_SIZE = os.environ.get('WHISPER_MODEL', 'small')

_model = None
_model_loaded = False
_model_lock = asyncio.Lock()


async def _load_model() -> None:
    global _model, _model_loaded
    async with _model_lock:
        if _model_loaded:
            return
        if STT_PROVIDER == 'whisper_local':
            import whisper

            loop = asyncio.get_event_loop()
            _model = await loop.run_in_executor(None, whisper.load_model, WHISPER_MODEL_SIZE)
        _model_loaded = True


@app.on_event('startup')
async def startup() -> None:
    asyncio.create_task(_load_model())


class TranscribeRequest(BaseModel):
    audio_bytes: str  # base64
    format: str = 'wav'  # webm | wav | mp3


class TranscribeResponse(BaseModel):
    text: str


@app.post('/transcribe', response_model=TranscribeResponse)
async def transcribe(req: TranscribeRequest) -> TranscribeResponse:
    audio_data = base64.b64decode(req.audio_bytes)
    provider = STT_PROVIDER

    if provider == 'openai':
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
        suffix = f'.{req.format}'
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_data)
            tmp_path = f.name
        try:
            with open(tmp_path, 'rb') as audio_file:
                transcript = await client.audio.transcriptions.create(model='whisper-1', file=audio_file)
            return TranscribeResponse(text=transcript.text)
        finally:
            os.unlink(tmp_path)

    # whisper_local
    if not _model_loaded:
        await _load_model()

    suffix = f'.{req.format}'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_data)
        tmp_path = f.name

    # Convert webm to wav using ffmpeg if needed
    wav_path = tmp_path
    if req.format == 'webm':
        wav_path = tmp_path.replace('.webm', '_converted.wav')
        proc = await asyncio.create_subprocess_exec(
            'ffmpeg',
            '-y',
            '-i',
            tmp_path,
            '-ar',
            '16000',
            '-ac',
            '1',
            wav_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: _model.transcribe(wav_path))
        return TranscribeResponse(text=result['text'].strip())
    finally:
        os.unlink(tmp_path)
        if wav_path != tmp_path and os.path.exists(wav_path):
            os.unlink(wav_path)


@app.get('/health')
async def health() -> dict:
    return {
        'ok': True,
        'provider': STT_PROVIDER,
        'model_loaded': _model_loaded,
    }
