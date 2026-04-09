import { useState, useRef, useCallback } from "react"
import SearchBar from "./components/SearchBar"
import ResultGrid from "./components/ResultGrid"
import StatsBar from "./components/StatsBar"
import VoiceButton from "./components/VoiceButton"

const API = import.meta.env.VITE_API_URL || "http://localhost:8000"

export default function App() {
  const [results, setResults] = useState([])
  const [query, setQuery] = useState("")
  const [queryType, setQueryType] = useState("")
  const [transcription, setTranscription] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [latency, setLatency] = useState(null)
  const [stats, setStats] = useState(null)

  // ── Search handlers ───────────────────────────────────────────────────────

  const searchText = useCallback(async (text) => {
    if (!text.trim()) return
    setLoading(true)
    setError("")
    setTranscription("")
    try {
      const res = await fetch(`${API}/search/text?q=${encodeURIComponent(text)}&k=12`)
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setResults(data.results)
      setQuery(data.query)
      setQueryType("text")
      setLatency({ total: data.latency_ms, encoder: data.encoder_latency_ms })
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  const searchImage = useCallback(async (file) => {
    setLoading(true)
    setError("")
    setTranscription("")
    const form = new FormData()
    form.append("file", file)
    try {
      const res = await fetch(`${API}/search/image?k=12`, { method: "POST", body: form })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setResults(data.results)
      setQuery("uploaded image")
      setQueryType("image")
      setLatency({ total: data.latency_ms, encoder: data.encoder_latency_ms })
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  const searchVoice = useCallback(async (audioBlob) => {
    setLoading(true)
    setError("")
    const form = new FormData()
    form.append("file", audioBlob, "voice.wav")
    try {
      const res = await fetch(`${API}/search/voice?k=12`, { method: "POST", body: form })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setResults(data.results)
      setQuery(data.query)
      setQueryType("voice")
      setTranscription(data.transcription || "")
      setLatency({ total: data.latency_ms, encoder: data.encoder_latency_ms, whisper: data.whisper_ms })
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }, [])

  const submitFeedback = useCallback(async (imagePath, vote) => {
    try {
      await fetch(`${API}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_path: imagePath, query, vote }),
      })
    } catch (e) {
      console.warn("Feedback failed:", e)
    }
  }, [query])

  return (
    <div className="app">
      <header className="header">
        <h1 className="logo">Visual Search</h1>
        <p className="tagline">Search images by text, voice, or image — powered by CLIP + FAISS</p>
      </header>

      <main className="main">
        <div className="search-area">
          <SearchBar onSearch={searchText} onImageUpload={searchImage} loading={loading} />
          <VoiceButton onResult={searchVoice} loading={loading} />
        </div>

        {transcription && (
          <div className="transcription">
            <span className="transcription-label">Heard:</span> "{transcription}"
          </div>
        )}

        {latency && (
          <StatsBar
            latency={latency}
            resultCount={results.length}
            queryType={queryType}
          />
        )}

        {error && <div className="error">{error}</div>}

        {loading && (
          <div className="loading">
            <div className="spinner" />
            <span>Searching{queryType === "voice" ? " (transcribing...)" : ""}...</span>
          </div>
        )}

        {!loading && results.length > 0 && (
          <ResultGrid results={results} onFeedback={submitFeedback} apiBase={API} />
        )}

        {!loading && results.length === 0 && !error && query && (
          <div className="empty">No results found for "{query}"</div>
        )}

        {!query && !loading && (
          <div className="hero-hint">
            <div className="hint-grid">
              {["dog running in rain", "mountain sunset", "busy city market", "rocket launch"].map(q => (
                <button key={q} className="hint-chip" onClick={() => searchText(q)}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
      </main>
    </div>
  )
}