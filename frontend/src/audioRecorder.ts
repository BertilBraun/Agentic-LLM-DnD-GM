export class AudioRecorder {
  private mediaRecorder: MediaRecorder | null = null
  private chunks: Blob[] = []

  async start(): Promise<void> {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    this.chunks = []
    this.mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' })
    this.mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) this.chunks.push(e.data)
    }
    this.mediaRecorder.start()
  }

  stop(): Promise<Blob> {
    return new Promise((resolve, reject) => {
      if (!this.mediaRecorder) return reject(new Error('Not recording'))
      this.mediaRecorder.onstop = () => {
        const blob = new Blob(this.chunks, { type: 'audio/webm' })
        this.mediaRecorder?.stream.getTracks().forEach((t) => t.stop())
        resolve(blob)
      }
      this.mediaRecorder.stop()
    })
  }
}
