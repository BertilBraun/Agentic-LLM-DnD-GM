interface Props { message: string; onDismiss?: () => void }
export function ErrorBanner({ message, onDismiss }: Props) {
  return (
    <div style={{ background: '#fee', border: '1px solid #f00', padding: '8px 12px', borderRadius: 4, margin: '8px 0', display: 'flex', justifyContent: 'space-between' }}>
      <span>{message}</span>
      {onDismiss && <button onClick={onDismiss} style={{ background: 'none', border: 'none', cursor: 'pointer', fontWeight: 'bold' }}>✕</button>}
    </div>
  )
}
