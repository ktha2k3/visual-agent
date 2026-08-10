import { useState, useRef, useEffect, useCallback, Component } from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeKatex from 'rehype-katex'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'

const AGENT_META = {
  supervisor:     { icon: '🎯', label: 'Supervisor',      color: '#818cf8' },
  research_agent: { icon: '🔬', label: 'Research Agent',  color: '#60a5fa' },
  vision_agent:   { icon: '👁️',  label: 'Vision Agent',   color: '#f59e0b' },
  agent:          { icon: '🧠', label: 'Agent Reasoning', color: '#60a5fa' },
  tools:          { icon: '🔧', label: 'Tool Execution',  color: '#34d399' },
}

function agentMeta(name) {
  return AGENT_META[name] ?? { icon: '🤖', label: name, color: '#94a3b8' }
}

const _LS_KEY = 'va_session_id'

function genSessionId() {
  return typeof crypto !== 'undefined' && crypto.randomUUID
    ? crypto.randomUUID()
    : Math.random().toString(36).slice(2) + Date.now().toString(36)
}

// ── Error boundary — prevents a broken KaTeX formula from crashing the app ──
class MarkdownErrorBoundary extends Component {
  state = { crashed: false }
  static getDerivedStateFromError() { return { crashed: true } }
  componentDidCatch(err) { console.warn('[markdown]', err.message) }
  render() {
    if (this.state.crashed) {
      // Reset on next content change so fixed text re-renders correctly
      return <span style={{ whiteSpace: 'pre-wrap' }}>{this.props.children}</span>
    }
    return this.props.children
  }
}

// Full markdown with math (for committed, complete messages)
function MarkdownContent({ children }) {
  return (
    <MarkdownErrorBoundary key={children}>
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
        {children}
      </ReactMarkdown>
    </MarkdownErrorBoundary>
  )
}

// Plain GFM only — used while streaming to avoid KaTeX crashing on partial LaTeX
function MarkdownStream({ children }) {
  return (
    <MarkdownErrorBoundary key={children}>
      <ReactMarkdown remarkPlugins={[remarkGfm]}>
        {children}
      </ReactMarkdown>
    </MarkdownErrorBoundary>
  )
}

// ── Agent card section ─────────────────────────────────────────────────────
function Section({ section }) {
  if (section.type === 'text') {
    return (
      <div className="sec-text">
        <MarkdownContent>{section.content}</MarkdownContent>
      </div>
    )
  }
  if (section.type === 'tool_call') {
    return (
      <div className="sec-tool-call">
        <span className="tool-badge call">⚡ {section.tool}</span>
        {section.args && <code className="tool-args">{section.args}</code>}
      </div>
    )
  }
  if (section.type === 'tool_result') {
    return (
      <div className="sec-tool-result">
        <span className="tool-badge result">📥 {section.tool} result</span>
        <pre className="tool-content">{section.content}</pre>
      </div>
    )
  }
  return null
}

function AgentCard({ name, sections, isActive }) {
  const bodyRef = useRef(null)
  const meta = agentMeta(name)
  useEffect(() => {
    if (bodyRef.current) bodyRef.current.scrollTop = bodyRef.current.scrollHeight
  }, [sections])
  return (
    <div className={`agent-card ${isActive ? 'active' : 'done'}`} style={{ '--accent': meta.color }}>
      <div className="agent-card-header">
        <span className="agent-icon">{meta.icon}</span>
        <span className="agent-name">{meta.label}</span>
        {isActive ? <span className="status-dot active" /> : <span className="status-done">✓</span>}
      </div>
      <div className="agent-card-body" ref={bodyRef}>
        {sections.map((s, i) => <Section key={i} section={s} />)}
        {isActive && sections.length === 0 && <span className="thinking">thinking…</span>}
      </div>
    </div>
  )
}

// ── Lightbox ───────────────────────────────────────────────────────────────
function Lightbox({ src, onClose }) {
  const [scale, setScale] = useState(1)

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose() }
    document.addEventListener('keydown', onKey)
    return () => document.removeEventListener('keydown', onKey)
  }, [onClose])

  const onWheel = (e) => {
    e.preventDefault()
    setScale(prev => Math.min(Math.max(prev - e.deltaY * 0.001, 0.3), 6))
  }

  return (
    <div className="lb-overlay" onClick={onClose} onWheel={onWheel}>
      <img
        src={src}
        alt="full size"
        className="lb-img"
        style={{ transform: `scale(${scale})` }}
        onClick={e => e.stopPropagation()}
        draggable={false}
      />
      <button className="lb-close" onClick={onClose} title="Close (Esc)">✕</button>
      <span className="lb-hint">Scroll to zoom · right-click to copy/save · Esc to close</span>
    </div>
  )
}

// ── Chat message bubble ────────────────────────────────────────────────────
function ChatMessage({ msg }) {
  const isUser = msg.role === 'user'
  const [lightbox, setLightbox] = useState(false)

  return (
    <div className={`chat-msg ${isUser ? 'user' : 'assistant'}`}>
      <div className="chat-avatar">{isUser ? '🧑' : '🤖'}</div>
      <div className="chat-bubble">
        {msg.imagePreview && (
          <img
            src={msg.imagePreview}
            alt="attached"
            className="chat-img-preview"
            onClick={() => setLightbox(true)}
            title="Click to view full size"
          />
        )}
        {isUser
          ? <p className="chat-text-user">{msg.text}</p>
          : (
            <div className="response-markdown">
              <MarkdownContent>{msg.text}</MarkdownContent>
            </div>
          )
        }
      </div>
      {lightbox && <Lightbox src={msg.imagePreview} onClose={() => setLightbox(false)} />}
    </div>
  )
}

// ── Main app ───────────────────────────────────────────────────────────────
export default function App() {
  // Persist session across page refreshes via localStorage
  const [sessionId, setSessionId] = useState(() => {
    const stored = localStorage.getItem(_LS_KEY)
    if (stored) return stored
    const id = genSessionId()
    localStorage.setItem(_LS_KEY, id)
    return id
  })
  const [input, setInput] = useState('')
  const [attachedImage, setAttachedImage] = useState(null)
  const [isUploading, setIsUploading] = useState(false)

  // Chat history: { role, text, imagePreview? }
  const [chatHistory, setChatHistory] = useState([])
  // Streaming accumulator for the current assistant reply
  const [streamingText, setStreamingText] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)

  // Agent activity for the current request
  const [agentData, setAgentData] = useState({})
  const [agentOrder, setAgentOrder] = useState([])

  const [errorMsg, setErrorMsg] = useState('')

  const chatEndRef = useRef(null)
  const fileInputRef = useRef(null)
  const agentTextAccumRef = useRef({})
  // Mirror of streamingText in a ref so the 'done' handler can read the
  // current value synchronously without calling setState inside a setState updater.
  const streamingTextRef = useRef('')

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [chatHistory, streamingText])

  // ── Image upload ───────────────────────────────────────────────────────────
  const uploadFile = useCallback(async (file) => {
    if (!file?.type.startsWith('image/')) return
    setIsUploading(true)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch('/api/upload', { method: 'POST', body: form })
      if (!res.ok) throw new Error(`Upload failed (${res.status})`)
      const { path } = await res.json()
      setAttachedImage({ previewUrl: URL.createObjectURL(file), serverPath: path, name: file.name || 'image' })
    } catch (err) {
      setErrorMsg(err.message)
    } finally {
      setIsUploading(false)
    }
  }, [])

  // Called by the × button — revokes the blob URL to free memory
  const removeImage = useCallback(() => {
    setAttachedImage(prev => { if (prev?.previewUrl) URL.revokeObjectURL(prev.previewUrl); return null })
  }, [])

  // Called on submit — clears the attachment WITHOUT revoking the blob URL
  // so the URL stored in chatHistory remains valid for display
  const clearImageAfterSend = useCallback(() => {
    setAttachedImage(null)
  }, [])

  const handleFileChange = (e) => {
    const file = e.target.files?.[0]
    if (file) uploadFile(file)
    e.target.value = ''
  }

  const handlePaste = useCallback((e) => {
    for (const item of e.clipboardData?.items ?? []) {
      if (item.type.startsWith('image/')) { e.preventDefault(); uploadFile(item.getAsFile()); break }
    }
  }, [uploadFile])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    const file = e.dataTransfer.files?.[0]
    if (file?.type.startsWith('image/')) uploadFile(file)
  }, [uploadFile])

  // ── Agent section helpers ──────────────────────────────────────────────────
  const pushSection = useCallback((agentName, section) => {
    if (section.type === 'text') {
      agentTextAccumRef.current[agentName] = (agentTextAccumRef.current[agentName] ?? '') + section.content
    }
    setAgentData(prev => {
      const agent = prev[agentName] ?? { sections: [], isActive: true }
      const sections = [...agent.sections]
      if (section.type === 'text') {
        const last = sections[sections.length - 1]
        if (last?.type === 'text') {
          sections[sections.length - 1] = { type: 'text', content: last.content + section.content }
          return { ...prev, [agentName]: { ...agent, sections } }
        }
      }
      return { ...prev, [agentName]: { ...agent, sections: [...sections, section] } }
    })
  }, [])

  // ── SSE handler ────────────────────────────────────────────────────────────
  const handleEvent = useCallback((event) => {
    switch (event.type) {
      case 'agent_start':
        setAgentOrder(prev => prev.includes(event.agent) ? prev : [...prev, event.agent])
        setAgentData(prev => ({ ...prev, [event.agent]: prev[event.agent] ?? { sections: [], isActive: true } }))
        break
      case 'token':
        if (event.agent === 'supervisor') {
          streamingTextRef.current += event.content
          setStreamingText(prev => prev + event.content)
        } else {
          pushSection(event.agent, { type: 'text', content: event.content })
        }
        break
      case 'tool_call':
        pushSection(event.agent, { type: 'tool_call', tool: event.tool, args: event.args })
        break
      case 'tool_result':
        pushSection(event.agent, { type: 'tool_result', tool: event.tool, content: event.content })
        break
      case 'handoff':
        // Only clear when the supervisor routes OUT to a sub-agent (transfer_to_X).
        // Do NOT clear on the return trip (transfer_back_to_supervisor).
        if (!event.tool?.toLowerCase().includes('back')) {
          streamingTextRef.current = ''
          setStreamingText('')
        }
        break
      case 'error':
        setErrorMsg(event.content)
        break
      case 'done': {
        // Mark all agents done
        setAgentData(prev => {
          const next = {}
          for (const [k, v] of Object.entries(prev)) next[k] = { ...v, isActive: false }
          return next
        })

        // Read current streaming text from the ref (synchronous, no stale closure)
        let final = streamingTextRef.current
        if (!final.trim() && !event.supervisor_responded) {
          const accum = agentTextAccumRef.current
          for (const name of ['agent', 'research_agent', 'vision_agent']) {
            if (accum[name]?.trim()) { final = accum[name]; break }
          }
          if (!final.trim()) {
            for (const [name, text] of Object.entries(accum)) {
              if (name !== 'tools' && text?.trim()) { final = text; break }
            }
          }
        }

        // Commit to chat history and clear display — separate setState calls,
        // never nested inside each other (calling setState inside a setState
        // updater is a React anti-pattern that silently drops the inner update).
        if (final.trim()) {
          setChatHistory(h => [...h, { role: 'assistant', text: final }])
        }
        streamingTextRef.current = ''
        setStreamingText('')
        break
      }
      default: break
    }
  }, [pushSection])

  // ── Submit ─────────────────────────────────────────────────────────────────
  const handleSubmit = async () => {
    const hasContent = input.trim() || attachedImage
    if (!hasContent || isStreaming || isUploading) return

    let message = input.trim()
    if (attachedImage) {
      message = message ? `${message}\n\nImage: ${attachedImage.serverPath}` : `Image: ${attachedImage.serverPath}`
    }

    // Add user message to history
    setChatHistory(h => [...h, {
      role: 'user',
      text: input.trim() || '(image)',
      imagePreview: attachedImage?.previewUrl,
    }])
    setInput('')
    clearImageAfterSend()  // don't revoke — URL is now owned by chatHistory

    // Reset agent activity for this turn
    setAgentData({})
    setAgentOrder([])
    streamingTextRef.current = ''
    setStreamingText('')
    setErrorMsg('')
    setIsStreaming(true)
    agentTextAccumRef.current = {}

    try {
      const response = await fetch('/api/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, session_id: sessionId }),
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try { handleEvent(JSON.parse(line.slice(6))) } catch { /* skip */ }
          }
        }
      }
    } catch (err) {
      setErrorMsg(err.message)
    }

    setIsStreaming(false)
  }

  const onKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit() }
  }

  const newChat = () => {
    // New UUID = new LangGraph thread = agent starts fresh
    const id = genSessionId()
    localStorage.setItem(_LS_KEY, id)
    setSessionId(id)
    setChatHistory([])
    setAgentData({})
    setAgentOrder([])
    setStreamingText('')
    setErrorMsg('')
  }

  const subAgents = agentOrder.filter(n => n !== 'supervisor')
  const canSend = (input.trim() || attachedImage) && !isStreaming && !isUploading

  return (
    <div className="layout">

      {/* Header */}
      <header className="topbar">
        <div className="topbar-brand">
          <span className="brand-icon">🤖</span>
          <span className="brand-name">Visual Agent System</span>
        </div>
        <div className="topbar-right">
          {isStreaming && (
            <div className="streaming-indicator">
              <span className="dot-pulse" />
              <span>Streaming</span>
            </div>
          )}
          <button className="new-chat-btn" onClick={newChat} title="New chat">
            ✏️ New chat
          </button>
        </div>
      </header>

      {/* Panels */}
      <div className="panels">

        {/* Left — Agent Activity */}
        <section className="panel panel-agents">
          <div className="panel-title">
            <span>Agent Activity</span>
            {subAgents.length > 0 && <span className="badge">{subAgents.length}</span>}
          </div>
          <div className="panel-body">
            {subAgents.length === 0
              ? <p className="empty">Agent activity streams here during inference…</p>
              : subAgents.map(name => (
                  <AgentCard key={name} name={name}
                    sections={agentData[name]?.sections ?? []}
                    isActive={agentData[name]?.isActive ?? false} />
                ))
            }
          </div>
        </section>

        {/* Right — Chat thread */}
        <section className="panel panel-chat">
          <div className="panel-title">
            <span>Conversation</span>
            {chatHistory.length > 0 && (
              <span className="badge">{Math.ceil(chatHistory.length / 2)} turn{chatHistory.length > 2 ? 's' : ''}</span>
            )}
          </div>
          <div className="panel-body chat-body">
            {chatHistory.length === 0 && !streamingText && !isStreaming && (
              <div className="chat-empty">
                <span className="chat-empty-icon">💬</span>
                <p>Start a conversation — ask a research question or attach an image.</p>
              </div>
            )}

            {chatHistory.map((msg, i) => <ChatMessage key={i} msg={msg} />)}

            {/* Streaming assistant reply */}
            {(isStreaming || streamingText) && (
              <div className="chat-msg assistant">
                <div className="chat-avatar">🤖</div>
                <div className="chat-bubble">
                  {streamingText
                    ? (
                      <div className="response-markdown">
                        <MarkdownStream>{streamingText}</MarkdownStream>
                        {isStreaming && <span className="cursor">▊</span>}
                      </div>
                    )
                    : <span className="thinking">thinking…</span>
                  }
                </div>
              </div>
            )}

            <div ref={chatEndRef} />
          </div>
        </section>
      </div>

      {/* Error banner */}
      {errorMsg && (
        <div className="error-banner">
          ⚠️ {errorMsg}
          <button className="error-close" onClick={() => setErrorMsg('')}>✕</button>
        </div>
      )}

      {/* Input bar */}
      <footer className="inputbar">
        {attachedImage && (
          <div className="image-preview-strip">
            <img src={attachedImage.previewUrl} alt="preview" className="preview-thumb" />
            <div className="preview-info">
              <span className="preview-name">{attachedImage.name}</span>
              <span className="preview-hint">will be sent with your message</span>
            </div>
            <button className="preview-remove" onClick={removeImage}>✕</button>
          </div>
        )}
        <div className="input-row">
          <button className="attach-btn" onClick={() => fileInputRef.current?.click()}
            disabled={isStreaming || isUploading} title="Attach image (or paste / drag-and-drop)">
            {isUploading ? <span className="spinner small" /> : (
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                <circle cx="8.5" cy="8.5" r="1.5" />
                <polyline points="21 15 16 10 5 21" />
              </svg>
            )}
          </button>
          <input ref={fileInputRef} type="file" accept="image/*" style={{ display: 'none' }} onChange={handleFileChange} />
          <textarea className="input-field" value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={onKeyDown} onPaste={handlePaste}
            onDrop={handleDrop} onDragOver={e => e.preventDefault()}
            placeholder={attachedImage ? 'Add a message or just send the image…' : 'Ask anything… follow-ups work too'}
            rows={2} disabled={isStreaming} />
          <button className="send-btn" onClick={handleSubmit} disabled={!canSend}>
            {isStreaming ? <span className="spinner" /> : (
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            )}
          </button>
        </div>
      </footer>
    </div>
  )
}
