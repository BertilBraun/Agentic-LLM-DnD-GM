import { useState, useRef } from 'react'
import { Spinner } from '../shared/Spinner'
import { AudioRecorder } from '../../audioRecorder'
import { api } from '../../api'
import { useGameStore } from '../../store'

interface Props {
  campaignId: string
  disabled?: boolean
}

export function AudioControls({ campaignId, disabled }: Props) {
  const [recording, setRecording] = useState(false)
  const [uploading, setUploading] = useState(false)
  const recorderRef = useRef(new AudioRecorder())
  const store = useGameStore()

  const startRecording = async () => {
    try {
      await recorderRef.current.start()
      setRecording(true)
    } catch (e) {
      store.setError('Microphone access denied')
    }
  }

  const stopAndSend = async () => {
    setRecording(false)
    setUploading(true)
    store.setAgentRunning(true)
    try {
      const blob = await recorderRef.current.stop()
      const { transcript } = await api.game.uploadAudio(campaignId, blob)
      store.appendMessage({ role: 'player', content: transcript })
      await api.game.sendMessage(campaignId, transcript)
    } catch (e) {
      store.setError((e as Error).message)
      store.setAgentRunning(false)
    } finally {
      setUploading(false)
    }
  }

  const [text, setText] = useState('')

  const sendText = async () => {
    if (!text.trim()) return
    const msg = text.trim()
    setText('')
    store.appendMessage({ role: 'player', content: msg })
    store.setAgentRunning(true)
    try {
      await api.game.sendMessage(campaignId, msg)
    } catch (e) {
      store.setError((e as Error).message)
      store.setAgentRunning(false)
    }
  }

  return (
    <div style={{ padding: '12px 16px', borderTop: '1px solid #e5e7eb', display: 'flex', flexDirection: 'column', gap: 8 }}>
      {disabled && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#6b7280', fontSize: 13 }}>
          <Spinner />
          <span>Waiting for response…</span>
        </div>
      )}
      <div style={{ display: 'flex', gap: 8 }}>
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && sendText()}
          placeholder="Type your action..."
          disabled={disabled || uploading}
          style={{ flex: 1, padding: '8px 12px', borderRadius: 6, border: '1px solid #d1d5db' }}
        />
        <button onClick={sendText} disabled={disabled || uploading || !text.trim()} style={{ padding: '8px 16px', cursor: 'pointer' }}>
          Send
        </button>
      </div>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        {!recording ? (
          <button onClick={startRecording} disabled={disabled || uploading} style={{ padding: '8px 16px', cursor: 'pointer', background: '#fee2e2', border: '1px solid #fca5a5', borderRadius: 6 }}>
            🎤 Record
          </button>
        ) : (
          <button onClick={stopAndSend} style={{ padding: '8px 16px', cursor: 'pointer', background: '#dc2626', color: '#fff', border: 'none', borderRadius: 6, animation: 'pulse 1s infinite' }}>
            ⏹ Stop & Send
          </button>
        )}
        {uploading && <Spinner />}
      </div>
    </div>
  )
}
