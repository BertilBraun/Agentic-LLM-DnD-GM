import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useGameStore } from './store'
import { api } from './api'

export function useSSE(campaignId: string) {
  const store = useGameStore()
  const navigate = useNavigate()

  useEffect(() => {
    const base = (import.meta.env.VITE_API_BASE_URL ?? '/api/v1').replace(/\/$/, '')
    const source = new EventSource(`${base}/campaigns/${campaignId}/stream`, { withCredentials: true })

    source.onmessage = (event) => {
      const payload = JSON.parse(event.data)
      switch (payload.type) {
        case 'ping':
          break
        case 'dm_text':
          store.appendMessage({ role: 'dm', content: payload.content })
          store.setAgentRunning(false)
          break
        case 'npc_speech':
          store.appendMessage({ role: 'npc', content: payload.content, npcName: payload.npc_name })
          store.setAgentRunning(false)
          break
        case 'npc_introduced':
          store.setActiveNPC({ npc_name: payload.npc_name, portrait_path: payload.portrait_path })
          if (payload.portrait_path) store.setCurrentImage(payload.portrait_path)
          store.setAgentRunning(false)
          break
        case 'npc_conversation_ended':
          store.clearActiveNPC()
          store.setAgentRunning(false)
          break
        case 'audio_ready':
          store.enqueueAudio(payload.stream_path)
          break
        case 'scene_ready':
          store.setCurrentImage(payload.file_path)
          break
        case 'portrait_ready':
          store.setCurrentImage(payload.file_path)
          break
        case 'phase_change':
          store.setPhase(payload.phase)
          if (payload.phase === 'active') {
            navigate(`/campaigns/${campaignId}/play`)
          }
          break
        case 'error':
          store.setError(payload.message)
          store.setAgentRunning(false)
          break
      }
    }

    source.onerror = () => {
      // EventSource will auto-reconnect; don't clear agentRunning on transient errors
    }

    return () => source.close()
  }, [campaignId])
}
