import { useEffect, useState } from 'react'
import { useGameStore } from '../../store'
import { api } from '../../api'
import { SceneImage } from './SceneImage'
import { TranscriptPanel } from './TranscriptPanel'
import { AudioControls } from './AudioControls'
import { NPCBadge } from './NPCBadge'
import { ErrorBanner } from '../shared/ErrorBanner'
import { Spinner } from '../shared/Spinner'

interface Props { campaignId: string }

export function GameView({ campaignId }: Props) {
  const store = useGameStore()
  const [audioPlaying, setAudioPlaying] = useState(false)

  // Drain the audio queue — play one clip at a time
  useEffect(() => {
    if (audioPlaying || store.audioQueue.length === 0) return
    const path = store.shiftAudio()
    if (!path) return
    setAudioPlaying(true)
    const audio = new Audio(api.mediaUrl(path))
    audio.onended = () => setAudioPlaying(false)
    audio.onerror = () => setAudioPlaying(false)
    audio.play().catch(() => setAudioPlaying(false))
  }, [audioPlaying, store.audioQueue.length])

  return (
    <div style={{ display: 'flex', height: '100vh', flexDirection: 'column' }}>
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        {/* Left: scene image */}
        <div style={{ width: '40%', padding: 16, display: 'flex', flexDirection: 'column', gap: 12, overflowY: 'auto', background: '#0f0f1a' }}>
          <SceneImage path={store.currentImage} />
          <NPCBadge npc={store.activeNPC} />
          {store.isAgentRunning && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, color: '#aaa' }}>
              <Spinner /> <span>DM is thinking...</span>
            </div>
          )}
        </div>
        {/* Right: transcript */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', background: '#1a1a2e', color: '#e5e7eb' }}>
          {store.error && (
            <ErrorBanner message={store.error} onDismiss={() => store.setError(null)} />
          )}
          <TranscriptPanel messages={store.messages} />
          <AudioControls campaignId={campaignId} disabled={store.isAgentRunning} />
        </div>
      </div>
    </div>
  )
}
