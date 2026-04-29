import { useGameStore } from '../../store'
import { CharacterStep } from './CharacterStep'
import { CampaignStep } from './CampaignStep'

interface Props { campaignId: string }

export function SetupWizard({ campaignId }: Props) {
  const phase = useGameStore((s) => s.phase)

  return (
    <div style={{ maxWidth: 720, margin: '0 auto', padding: 24 }}>
      <div style={{ display: 'flex', gap: 16, marginBottom: 24 }}>
        <StepIndicator label="1. Character Creation" active={phase === 'character_creation'} done={phase !== 'character_creation'} />
        <StepIndicator label="2. Campaign Design" active={phase === 'campaign_design'} done={phase === 'active'} />
      </div>
      {phase === 'character_creation' && <CharacterStep campaignId={campaignId} />}
      {phase === 'campaign_design' && <CampaignStep campaignId={campaignId} />}
    </div>
  )
}

function StepIndicator({ label, active, done }: { label: string; active: boolean; done: boolean }) {
  const bg = done ? '#d1fae5' : active ? '#ede9fe' : '#f3f4f6'
  const color = done ? '#065f46' : active ? '#5b21b6' : '#6b7280'
  return (
    <div style={{ flex: 1, padding: '8px 12px', borderRadius: 6, background: bg, color, fontWeight: active ? 600 : 400 }}>
      {done ? '✓ ' : ''}{label}
    </div>
  )
}
