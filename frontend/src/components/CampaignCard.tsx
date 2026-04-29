import { useNavigate } from 'react-router-dom'
import type { Campaign } from '../api'
import { api } from '../api'

interface Props {
  campaign: Campaign
  onDeleted: () => void
}

const PHASE_LABELS: Record<string, string> = {
  character_creation: '🧙 Character Creation',
  campaign_design: '📜 Campaign Design',
  active: '⚔️ Active',
  completed: '✅ Completed',
}

export function CampaignCard({ campaign, onDeleted }: Props) {
  const navigate = useNavigate()

  const handleContinue = () => {
    if (campaign.phase === 'active' || campaign.phase === 'completed') {
      navigate(`/campaigns/${campaign.id}/play`)
    } else {
      navigate(`/campaigns/${campaign.id}/setup`)
    }
  }

  const handleDelete = async () => {
    if (!confirm(`Delete "${campaign.title}"?`)) return
    await api.campaigns.delete(campaign.id)
    onDeleted()
  }

  return (
    <div style={{ border: '1px solid #ddd', borderRadius: 8, padding: 16, marginBottom: 12, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <div>
        <div style={{ fontWeight: 'bold', fontSize: 16 }}>{campaign.title}</div>
        <div style={{ color: '#666', fontSize: 13 }}>{PHASE_LABELS[campaign.phase] ?? campaign.phase}</div>
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        <button onClick={handleContinue} style={{ padding: '6px 14px', cursor: 'pointer' }}>
          {campaign.phase === 'completed' ? 'View' : 'Continue'}
        </button>
        <button onClick={handleDelete} style={{ padding: '6px 14px', cursor: 'pointer', background: '#fee', border: '1px solid #faa' }}>
          Delete
        </button>
      </div>
    </div>
  )
}
