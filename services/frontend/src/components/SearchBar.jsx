import { useState, useRef } from "react"

export default function SearchBar({ onSearch, onImageUpload, loading }) {
  const [text, setText] = useState("")
  const fileRef = useRef()

  const handleSubmit = (e) => {
    e.preventDefault()
    if (text.trim()) onSearch(text.trim())
  }

  const handleFile = (e) => {
    const file = e.target.files?.[0]
    if (file) onImageUpload(file)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    const file = e.dataTransfer.files?.[0]
    if (file && file.type.startsWith("image/")) onImageUpload(file)
  }

  return (
    <form className="search-bar" onSubmit={handleSubmit}
      onDragOver={e => e.preventDefault()} onDrop={handleDrop}>
      <input
        className="search-input"
        type="text"
        placeholder="Search images... (or drag & drop an image)"
        value={text}
        onChange={e => setText(e.target.value)}
        disabled={loading}
      />
      <button type="submit" className="search-btn" disabled={loading || !text.trim()}>
        Search
      </button>
      <button
        type="button"
        className="upload-btn"
        onClick={() => fileRef.current?.click()}
        disabled={loading}
        title="Search by image"
      >
        Upload
      </button>
      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={handleFile}
      />
    </form>
  )
}