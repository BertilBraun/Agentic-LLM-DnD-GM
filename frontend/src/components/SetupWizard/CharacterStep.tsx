import { useGameStore } from '../../store'
import { TranscriptPanel } from '../GameView/TranscriptPanel'
import { AudioControls } from '../GameView/AudioControls'
import { SceneImage } from '../GameView/SceneImage'
import { ErrorBanner } from '../shared/ErrorBanner'

interface Props { campaignId: string }

export function CharacterStep({ campaignId }: Props) {
  const store = useGameStore()
  return (
    <div>
      <h2 style={{ marginBottom: 8 }}>Character Creation</h2>
      <p style={{ color: '#6b7280', marginBottom: 16 }}>Answer the DM's questions to build your character.</p>
      {store.currentImage && (
        <div style={{ marginBottom: 16, maxWidth: 300 }}>
          <SceneImage path={store.currentImage} />
        </div>
      )}
      {store.error && <ErrorBanner message={store.error} onDismiss={() => store.setError(null)} />}
      <div style={{ border: '1px solid #e5e7eb', borderRadius: 8, display: 'flex', flexDirection: 'column', height: 400 }}>
        <TranscriptPanel messages={store.messages} />
        <AudioControls campaignId={campaignId} disabled={store.isAgentRunning} />
      </div>
    </div>
  )
}
