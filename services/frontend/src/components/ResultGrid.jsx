import { useState } from "react"

export function ResultGrid({ results, onFeedback, apiBase }) {
  const [votes, setVotes] = useState({})

  const vote = (path, v) => {
    setVotes(prev => ({ ...prev, [path]: v }))
    onFeedback(path, v)
  }

  return (
    <div className="result-grid">
      {results.map((r) => (
        <div key={r.path} className="result-card">
          <div className="result-img-wrap">
            <img
              src={`${apiBase}${r.url}`}
              alt={r.category}
              className="result-img"
              loading="lazy"
              onError={e => { e.target.style.display = "none" }}
            />
          </div>
          <div className="result-meta">
            <span className="result-category">{r.category.replace(/_/g, " ")}</span>
            <span className="result-score">{(r.score * 100).toFixed(1)}%</span>
          </div>
          <div className="result-actions">
            <button
              className={`vote-btn ${votes[r.path] === 1 ? "voted-up" : ""}`}
              onClick={() => vote(r.path, 1)}
              title="Relevant"
            >+</button>
            <button
              className={`vote-btn ${votes[r.path] === -1 ? "voted-down" : ""}`}
              onClick={() => vote(r.path, -1)}
              title="Not relevant"
            >−</button>
          </div>
        </div>
      ))}
    </div>
  )
}

export default ResultGrid

export function StatsBar({ latency, resultCount, queryType }) {
  const icons = { text: "T", image: "I", voice: "V" }
  return (
    <div className="stats-bar">
      <span className="stat">
        <span className="stat-label">type</span>
        <span className="stat-value">{icons[queryType] || "?"} {queryType}</span>
      </span>
      <span className="stat">
        <span className="stat-label">results</span>
        <span className="stat-value">{resultCount}</span>
      </span>
      <span className="stat">
        <span className="stat-label">total</span>
        <span className="stat-value">{latency.total?.toFixed(0)}ms</span>
      </span>
      <span className="stat">
        <span className="stat-label">encoder</span>
        <span className="stat-value">{latency.encoder?.toFixed(0)}ms</span>
      </span>
      {latency.whisper && (
        <span className="stat">
          <span className="stat-label">whisper</span>
          <span className="stat-value">{latency.whisper?.toFixed(0)}ms</span>
        </span>
      )}
    </div>
  )
}