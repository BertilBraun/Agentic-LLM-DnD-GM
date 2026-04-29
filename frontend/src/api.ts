const BASE = import.meta.env.VITE_API_BASE_URL ?? '/api/v1'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    credentials: 'include',
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? res.statusText)
  }
  return res.json()
}

export const api = {
  auth: {
    register: (email: string, password: string) =>
      request('/auth/register', { method: 'POST', body: JSON.stringify({ email, password }) }),
    login: (email: string, password: string) =>
      request<{ access_token: string }>('/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) }),
  },
  campaigns: {
    list: () => request<Campaign[]>('/campaigns'),
    create: (title: string, language = 'en') =>
      request<Campaign>('/campaigns', { method: 'POST', body: JSON.stringify({ title, language }) }),
    get: (id: string) => request<CampaignDetail>(`/campaigns/${id}`),
    delete: (id: string) => request(`/campaigns/${id}`, { method: 'DELETE' }),
  },
  game: {
    uploadAudio: (campaignId: string, blob: Blob) => {
      const fd = new FormData()
      fd.append('file', blob, 'recording.webm')
      return request<{ file_path: string; transcript: string }>(`/campaigns/${campaignId}/audio`, {
        method: 'POST',
        headers: {},
        body: fd,
      })
    },
    sendMessage: (campaignId: string, content: string) =>
      request(`/campaigns/${campaignId}/message`, { method: 'POST', body: JSON.stringify({ content }) }),
    getTurns: (campaignId: string, limit = 50, playOnly = false) =>
      request<Turn[]>(`/campaigns/${campaignId}/turns?limit=${limit}${playOnly ? '&play_only=true' : ''}`),
  },
  mediaUrl: (path: string) => `${BASE}/media/${path}`,
}

export interface Campaign {
  id: string
  title: string
  phase: string
  created_at: string
}

export interface CampaignDetail extends Campaign {
  language: string
  plan_json: Record<string, unknown> | null
  visual_style: string | null
  updated_at: string
}

export interface Turn {
  id: string
  role: 'player' | 'dm' | 'npc' | 'system'
  content: string
  npc_name: string | null
  audio_path: string | null
  image_path: string | null
  created_at: string
}
