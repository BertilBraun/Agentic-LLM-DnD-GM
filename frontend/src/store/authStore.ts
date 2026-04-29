import { create } from 'zustand'

interface AuthState {
  token: string | null
  userId: string | null
  email: string | null
  /** True while we're attempting a silent refresh on page load */
  isLoading: boolean
  login: (token: string) => void
  logout: () => void
  finishLoading: () => void
}

function parseJwt(token: string): { sub: string; exp: number; email?: string } | null {
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
  const isExpired = payload ? payload.exp * 1000 < Date.now() : true

  // If we had a token but it's expired, we might still have a valid refresh cookie.
  // Set isLoading so ProtectedRoute can attempt a silent refresh before redirecting.
  const isLoading = !!saved && isExpired

  return {
    token: saved && !isExpired ? saved : null,
    userId: saved && !isExpired ? (payload?.sub ?? null) : null,
    email: null,
    isLoading,

    login(token) {
      localStorage.setItem('access_token', token)
      const p = parseJwt(token)
      set({ token, userId: p?.sub ?? null, isLoading: false })
    },

    logout() {
      localStorage.removeItem('access_token')
      set({ token: null, userId: null, email: null, isLoading: false })
    },

    finishLoading() {
      set({ isLoading: false })
    },
  }
})
