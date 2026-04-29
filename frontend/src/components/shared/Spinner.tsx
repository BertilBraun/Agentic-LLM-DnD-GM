export function Spinner() {
  return (
    <>
      <style>{`@keyframes spinner-spin { to { transform: rotate(360deg); } }`}</style>
      <div style={{ display: 'inline-block', width: 20, height: 20, border: '3px solid #ccc', borderTopColor: '#555', borderRadius: '50%', animation: 'spinner-spin 0.7s linear infinite' }} />
    </>
  )
}
