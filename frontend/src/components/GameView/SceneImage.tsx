import { api } from '../../api'

interface Props { path: string | null }

export function SceneImage({ path }: Props) {
  if (!path) return <div style={{ width: '100%', aspectRatio: '16/9', background: '#1a1a2e', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#444', borderRadius: 8 }}>No scene yet</div>
  return <img src={api.mediaUrl(path)} alt="Scene" style={{ width: '100%', borderRadius: 8, objectFit: 'cover' }} />
}
