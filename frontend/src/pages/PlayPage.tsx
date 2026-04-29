import { useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { useGameStore } from '../store'
import { useSSE } from '../sse'
import { GameView } from '../components/GameView/GameView'
import { api } from '../api'

const OPENING_SENTINEL = '__opening_scene__'

export function PlayPage() {
  const { id = '' } = useParams<{ id: string }>()
  const store = useGameStore()

  useSSE(id)

  useEffect(() => {
    store.reset()
    store.setPhase('active')
    api.game.getTurns(id, 50, true).then((turns) => {
      for (const t of turns) {
        if (t.role !== 'system') {
          store.appendMessage({ role: t.role, content: t.content, npcName: t.npc_name ?? undefined })
        }
        if (t.image_path) store.setCurrentImage(t.image_path)
      }
      // Trigger opening scene if no turns yet
      if (turns.length === 0) {
        api.game.sendMessage(id, OPENING_SENTINEL).catch(() => {})
        store.setAgentRunning(true)
      }
    })
  }, [id])

  return <GameView campaignId={id} />
}
