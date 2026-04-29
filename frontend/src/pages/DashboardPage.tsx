import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, type Campaign } from '../api'
import { useAuthStore } from '../store'
import { CampaignCard } from '../components/CampaignCard'
import { Spinner } from '../components/shared/Spinner'

export function DashboardPage() {
  const [campaigns, setCampaigns] = useState<Campaign[]>([])
  const [loading, setLoading] = useState(true)
  const logout = useAuthStore((s) => s.logout)
  const navigate = useNavigate()

  const handleLogout = async () => {
    await api.auth.logout().catch(() => {})
    logout()
    navigate('/login')
  }

  const load = async () => {
    setLoading(true)
    try {
      setCampaigns(await api.campaigns.list())
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const newCampaign = async () => {
    const title = prompt('Campaign title:') || 'Untitled Campaign'
    const c = await api.campaigns.create(title)
    navigate(`/campaigns/${c.id}/setup`)
  }

  return (
    <div style={{ maxWidth: 700, margin: '0 auto', padding: 24 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <h1>⚔️ Your Campaigns</h1>
        <div style={{ display: 'flex', gap: 8 }}>
          <button onClick={newCampaign} style={{ padding: '8px 16px', cursor: 'pointer', background: '#5b21b6', color: '#fff', border: 'none', borderRadius: 6 }}>
            + New Campaign
          </button>
          <button onClick={handleLogout} style={{ padding: '8px 16px', cursor: 'pointer' }}>Log out</button>
        </div>
      </div>
      {loading ? <Spinner /> : campaigns.length === 0 ? (
        <p style={{ color: '#9ca3af', textAlign: 'center', marginTop: 48 }}>No campaigns yet. Start one!</p>
      ) : (
        campaigns.map((c) => <CampaignCard key={c.id} campaign={c} onDeleted={load} />)
      )}
    </div>
  )
}
