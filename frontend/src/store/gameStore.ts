import { create } from 'zustand'

export interface Message {
  id: string
  role: 'player' | 'dm' | 'npc' | 'system'
  content: string
  npcName?: string
}

export interface NPC {
  npc_name: string
  portrait_path?: string
}

interface GameState {
  messages: Message[]
  currentImage: string | null
  audioQueue: string[]
  activeNPC: NPC | null
  phase: string
  isAgentRunning: boolean
  error: string | null

  appendMessage: (msg: Omit<Message, 'id'>) => void
  setCurrentImage: (path: string) => void
  enqueueAudio: (path: string) => void
  shiftAudio: () => string | undefined
  setActiveNPC: (npc: NPC) => void
  clearActiveNPC: () => void
  setPhase: (phase: string) => void
  setAgentRunning: (v: boolean) => void
  setError: (msg: string | null) => void
  reset: () => void
}

export const useGameStore = create<GameState>((set, get) => ({
  messages: [],
  currentImage: null,
  audioQueue: [],
  activeNPC: null,
  phase: 'character_creation',
  isAgentRunning: false,
  error: null,

  appendMessage: (msg) =>
    set((s) => ({
      messages: [...s.messages, { ...msg, id: crypto.randomUUID() }],
    })),
  setCurrentImage: (path) => set({ currentImage: path }),
  enqueueAudio: (path) => set((s) => ({ audioQueue: [...s.audioQueue, path] })),
  shiftAudio: () => {
    const [next, ...rest] = get().audioQueue
    set({ audioQueue: rest })
    return next
  },
  setActiveNPC: (npc) => set({ activeNPC: npc }),
  clearActiveNPC: () => set({ activeNPC: null }),
  setPhase: (phase) => set({ phase }),
  setAgentRunning: (v) => set({ isAgentRunning: v }),
  setError: (msg) => set({ error: msg }),
  reset: () =>
    set({
      messages: [],
      currentImage: null,
      audioQueue: [],
      activeNPC: null,
      phase: 'character_creation',
      isAgentRunning: false,
      error: null,
    }),
}))
