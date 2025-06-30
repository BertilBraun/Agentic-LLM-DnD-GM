"""Web dashboard for voice-first LLM-DnD gameplay.

This initial implementation provides:
    • FastAPI server with root HTML route.
    • WebSocket endpoint `/ws/audio` for microphone streaming (placeholder – echoes bytes length).
    • Static HTML/JS capturing microphone via MediaRecorder and streaming Opus chunks.

Future work (outside current scope):
    – Hook STT backend for real Whisper transcription.
    – Display live transcripts and generated images.
    – Playback TTS audio responses.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect  # type: ignore
from fastapi.responses import HTMLResponse  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore

app = FastAPI(title="LLM-DnD Dashboard")

# ---------------------------------------------------------------------------
# Frontend HTML (inlined for simplicity)
# ---------------------------------------------------------------------------

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>LLM-DnD Dashboard</title>
  <style>
    body { font-family: sans-serif; margin: 2rem; }
    #status { margin-bottom: 1rem; color: green; }
    button { padding: 0.5rem 1rem; }
  </style>
</head>
<body>
  <h1>LLM-DnD Voice Dashboard</h1>
  <div id="status">Idle</div>
  <button id="recordBtn">Start Recording</button>
  <script>
    const statusEl = document.getElementById('status');
    const btn = document.getElementById('recordBtn');
    let ws = null;
    let mediaRecorder = null;

    async function init() {
      ws = new WebSocket(`ws://${location.host}/ws/audio`);
      ws.onopen = () => statusEl.textContent = 'WebSocket connected';
      ws.onclose = () => statusEl.textContent = 'WebSocket closed';
      ws.onerror = (e) => console.error(e);
      ws.onmessage = (ev) => console.log('Server:', ev.data);
    }

    async function startRecording() {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
      mediaRecorder.ondataavailable = (e) => {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(e.data);
        }
      };
      mediaRecorder.start(250); // send chunks every 250ms
      statusEl.textContent = 'Recording… (sending audio)';
      btn.textContent = 'Stop Recording';
    }

    btn.addEventListener('click', async () => {
      if (!ws) await init();
      if (mediaRecorder && mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
        statusEl.textContent = 'Idle';
        btn.textContent = 'Start Recording';
      } else {
        await startRecording();
      }
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return HTML_PAGE


@app.websocket("/ws/audio")
async def audio_ws(ws: WebSocket) -> None:  # pragma: no cover
    await ws.accept()
    try:
        while True:
            chunk = await ws.receive_bytes()
            # Placeholder: just acknowledge size
            await ws.send_text(f"received {len(chunk)} bytes")
    except WebSocketDisconnect:
        pass


# Convenience for `uvicorn -m src.ui.dashboard:app`
__all__ = ["app"] 