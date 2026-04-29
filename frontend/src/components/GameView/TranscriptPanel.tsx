import { useEffect, useRef } from 'react'
import type { Message } from '../../store/gameStore'

interface Props { messages: Message[] }

const ROLE_STYLES: Record<string, { label: string; color: string }> = {
  player: { label: 'You', color: '#2563eb' },
  dm: { label: 'DM', color: '#7c3aed' },
  npc: { label: 'NPC', color: '#059669' },
  system: { label: '—', color: '#9ca3af' },
}

export function TranscriptPanel({ messages }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  return (
    <div style={{ flex: 1, overflowY: 'auto', padding: '12px 16px', display: 'flex', flexDirection: 'column', gap: 12 }}>
      {messages.map((m) => {
        const style = ROLE_STYLES[m.role] ?? ROLE_STYLES.system
        return (
          <div key={m.id}>
            <span style={{ fontWeight: 'bold', color: style.color, marginRight: 8 }}>
              {m.npcName ?? style.label}:
            </span>
            <span>{m.content}</span>
          </div>
        )
      })}
      <div ref={bottomRef} />
    </div>
  )
}
