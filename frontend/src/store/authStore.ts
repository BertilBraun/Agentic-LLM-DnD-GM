import { create } from 'zustand'

interface AuthState {
  token: string | null
  userId: string | null
  email: string | null
  login: (token: string) => void
  logout: () => void
}

function parseJwt(token: string): { sub: string; email?: string } | null {
  try {
    const [, payload] = token.split('.')
    return JSON.parse(atob(payload.replace(/-/g, '+').replace(/_/g, '/')))
  } catch {
    return null
  }
}

export const useAuthStore = create<AuthState>((set) => {
  const saved = localStorage.getItem('access_token')
  const payload = saved ? parseJwt(saved) : null
  const isExpired = payload ? (payload as any).exp * 1000 < Date.now() : true

  return {
    token: saved && !isExpired ? saved : null,
    userId: saved && !isExpired ? (payload?.sub ?? null) : null,
    email: null,

    login(token) {
      localStorage.setItem('access_token', token)
      const p = parseJwt(token)
      set({ token, userId: p?.sub ?? null })
    },

    logout() {
      localStorage.removeItem('access_token')
      set({ token: null, userId: null, email: null })
    },
  }
})
