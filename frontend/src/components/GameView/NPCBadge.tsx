import type { NPC } from '../../store/gameStore'
import { api } from '../../api'

interface Props { npc: NPC | null }

export function NPCBadge({ npc }: Props) {
  if (!npc) return null
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 10px', background: '#f0fdf4', border: '1px solid #86efac', borderRadius: 20 }}>
      {npc.portrait_path && (
        <img src={api.mediaUrl(npc.portrait_path)} alt={npc.npc_name} style={{ width: 28, height: 28, borderRadius: '50%', objectFit: 'cover' }} />
      )}
      <span style={{ fontWeight: 600, color: '#166534' }}>🗣 {npc.npc_name}</span>
    </div>
  )
}
