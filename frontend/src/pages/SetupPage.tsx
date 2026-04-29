import { useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { useGameStore } from '../store'
import { useSSE } from '../sse'
import { SetupWizard } from '../components/SetupWizard/SetupWizard'
import { api } from '../api'

export function SetupPage() {
  const { id = '' } = useParams<{ id: string }>()
  const store = useGameStore()

  useSSE(id)

  useEffect(() => {
    store.reset()
    // Load existing turns and set initial phase
    api.campaigns.get(id).then((c) => {
      store.setPhase(c.phase)
    })
    api.game.getTurns(id, 50).then((turns) => {
      for (const t of turns) {
        if (t.role !== 'system') {
          store.appendMessage({ role: t.role, content: t.content, npcName: t.npc_name ?? undefined })
        }
      }
      // If no turns yet, send a first message to start character creation
      if (turns.length === 0) {
        api.game.sendMessage(id, 'Hello, I want to create a character.').catch(() => {})
        store.setAgentRunning(true)
      }
    })
  }, [id])

  return <SetupWizard campaignId={id} />
}
