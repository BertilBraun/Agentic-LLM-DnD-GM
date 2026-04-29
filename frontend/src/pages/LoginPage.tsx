import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api } from '../api'
import { useAuthStore } from '../store'
import { ErrorBanner } from '../components/shared/ErrorBanner'

export function LoginPage() {
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const login = useAuthStore((s) => s.login)
  const navigate = useNavigate()

  const submit = async () => {
    setError(null)
    setLoading(true)
    try {
      if (mode === 'register') {
        await api.auth.register(email, password)
      }
      const { access_token } = await api.auth.login(email, password)
      login(access_token)
      navigate('/dashboard')
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ maxWidth: 380, margin: '80px auto', padding: 24, border: '1px solid #e5e7eb', borderRadius: 12 }}>
      <h1 style={{ marginBottom: 24, textAlign: 'center' }}>⚔️ LLM-DnD</h1>
      {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <input placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} type="email" style={{ padding: '8px 12px', borderRadius: 6, border: '1px solid #d1d5db' }} />
        <input placeholder="Password" value={password} onChange={(e) => setPassword(e.target.value)} type="password" onKeyDown={(e) => e.key === 'Enter' && submit()} style={{ padding: '8px 12px', borderRadius: 6, border: '1px solid #d1d5db' }} />
        <button onClick={submit} disabled={loading} style={{ padding: '10px', cursor: 'pointer', background: '#5b21b6', color: '#fff', border: 'none', borderRadius: 6, fontWeight: 600 }}>
          {loading ? 'Loading...' : mode === 'login' ? 'Log In' : 'Register'}
        </button>
        <button onClick={() => setMode(mode === 'login' ? 'register' : 'login')} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#5b21b6', textDecoration: 'underline' }}>
          {mode === 'login' ? 'Create account' : 'Back to login'}
        </button>
      </div>
    </div>
  )
}
