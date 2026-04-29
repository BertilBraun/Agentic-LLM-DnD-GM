import { useEffect } from 'react'
import { Navigate, Outlet } from 'react-router-dom'
import { api } from '../api'
import { useAuthStore } from '../store'

export function ProtectedRoute() {
  const token = useAuthStore((s) => s.token)
  const isLoading = useAuthStore((s) => s.isLoading)
  const login = useAuthStore((s) => s.login)
  const finishLoading = useAuthStore((s) => s.finishLoading)

  useEffect(() => {
    if (!isLoading) return
    api.auth.refresh()
      .then(({ access_token }) => login(access_token))
      .catch(() => finishLoading())
  }, [isLoading])

  if (isLoading) return null  // blank screen while refreshing (brief)
  return token ? <Outlet /> : <Navigate to="/login" replace />
}
