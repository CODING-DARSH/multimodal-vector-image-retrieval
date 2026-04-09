import { useState, useRef } from "react"

/*
WHY THIS COMPONENT EXISTS:
  The Web MediaRecorder API lets us record audio from the microphone.
  We record while button is held down, stop on release.
  Send the recorded blob as WAV to /search/voice.

  Browser compatibility note:
  - Chrome: records as webm/opus by default
  - Safari: records as mp4/aac
  Whisper handles both formats natively so we don't need to convert.
*/

export default function VoiceButton({ onResult, loading }) {
  const [recording, setRecording] = useState(false)
  const [supported, setSupported] = useState(!!navigator.mediaDevices?.getUserMedia)
  const mediaRef = useRef(null)
  const chunksRef = useRef([])

  const startRecording = async () => {
    if (recording || loading) return
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const recorder = new MediaRecorder(stream)
      chunksRef.current = []

      recorder.ondataavailable = e => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }

      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/wav" })
        stream.getTracks().forEach(t => t.stop())
        onResult(blob)
      }

      recorder.start()
      mediaRef.current = recorder
      setRecording(true)
    } catch (e) {
      console.error("Mic access denied:", e)
      setSupported(false)
    }
  }

  const stopRecording = () => {
    if (mediaRef.current && recording) {
      mediaRef.current.stop()
      setRecording(false)
    }
  }

  if (!supported) return null

  return (
    <button
      className={`voice-btn ${recording ? "recording" : ""}`}
      onMouseDown={startRecording}
      onMouseUp={stopRecording}
      onTouchStart={startRecording}
      onTouchEnd={stopRecording}
      disabled={loading && !recording}
      title="Hold to record voice search"
    >
      {recording ? "● Release to search" : "Hold to speak"}
    </button>
  )
}