import { useCallback, useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import rehypeKatex from 'rehype-katex'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import { accessToken, authHeaders, supabase } from './supabase'

const SUBJECTS = [
  { id: 'math', label: 'Toán', icon: '∫' },
  { id: 'physics', label: 'Vật lí', icon: '⚛' },
  { id: 'chemistry', label: 'Hóa học', icon: '⚗' },
  { id: 'biology', label: 'Sinh học', icon: '🧬' },
]
const SUBJECT_LABEL = Object.fromEntries(SUBJECTS.map(item => [item.id, item.label]))
const SESSION_KEY = 'stem_tutor_session'
const KNOWLEDGE_SELECTION_KEY = 'stem_tutor_knowledge_selection'
const MAX_UPLOAD_BYTES = 25 * 1024 * 1024

function assertUploadSize(file) {
  if (file?.size > MAX_UPLOAD_BYTES) throw new Error('Tệp vượt quá giới hạn 25 MB.')
}

function sessionId() {
  const existing = localStorage.getItem(SESSION_KEY)
  if (existing) return existing
  const value = crypto.randomUUID()
  localStorage.setItem(SESSION_KEY, value)
  return value
}

async function apiFetch(path, options = {}) {
  const headers = { ...(await authHeaders()), ...(options.headers || {}) }
  const response = await fetch(path, { ...options, headers })
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}))
    throw new Error(payload.detail || `HTTP ${response.status}`)
  }
  return response
}

function websocketUrl(path) {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}${path}`
}

async function streamTutorWebSocket(request, onEvent) {
  const token = await accessToken()
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(websocketUrl('/api/tutor/ws'))
    let settled = false
    let receivedEvent = false
    const fail = (message) => {
      if (settled) return
      settled = true
      const error = new Error(message)
      error.canFallback = !receivedEvent
      reject(error)
    }
    socket.onopen = () => socket.send(JSON.stringify({ access_token: token, request }))
    socket.onmessage = (message) => {
      receivedEvent = true
      try {
        const event = JSON.parse(message.data)
        onEvent(event)
        if (event.type === 'done' && !settled) {
          settled = true
          resolve()
          socket.close()
        }
      } catch (_error) {
        fail('WebSocket returned an invalid event')
      }
    }
    socket.onerror = () => fail('WebSocket connection failed')
    socket.onclose = () => {
      if (!settled) fail('WebSocket closed before the response completed')
    }
  })
}

async function streamTutorSse(request, onEvent) {
  const response = await apiFetch('/api/tutor/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  })
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const blocks = buffer.split('\n\n')
    buffer = blocks.pop()
    for (const block of blocks) {
      const line = block.split('\n').find(item => item.startsWith('data: '))
      if (line) onEvent(JSON.parse(line.slice(6)))
    }
  }
}

function Markdown({ children }) {
  return <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>{children || ''}</ReactMarkdown>
}

function AuthenticatedVideo({ visualization }) {
  const [source, setSource] = useState('')
  const [error, setError] = useState('')
  useEffect(() => {
    let active = true
    let objectUrl = ''
    apiFetch(visualization.video_url)
      .then(response => response.blob())
      .then(blob => {
        if (!active) return
        objectUrl = URL.createObjectURL(blob)
        setSource(objectUrl)
      })
      .catch(err => { if (active) setError(err.message) })
    return () => {
      active = false
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [visualization.video_url])
  return <section className="visualization-card">
    <header><div><span className="eyebrow">Animation Manim</span><h3>{visualization.concept}</h3></div>{visualization.cached && <small>Đã dùng lại</small>}</header>
    {source && <video controls playsInline preload="metadata" src={source}>Trình duyệt không hỗ trợ video.</video>}
    {!source && !error && <div className="visualization-loading">Đang tải animation…</div>}
    {error && <p className="visualization-error">Không thể tải animation: {error}</p>}
    {visualization.formulas?.length > 0 && <Markdown>{visualization.formulas.map(formula => `$$${formula}$$`).join('\n\n')}</Markdown>}
    <p>{visualization.key_takeaway}</p>
  </section>
}

function Login() {
  const [email, setEmail] = useState('')
  const [status, setStatus] = useState('')
  const sendLink = async (event) => {
    event.preventDefault()
    setStatus('Đang gửi liên kết…')
    const { error } = await supabase.auth.signInWithOtp({
      email, options: { emailRedirectTo: window.location.origin },
    })
    setStatus(error ? error.message : 'Đã gửi! Hãy kiểm tra email của bạn.')
  }
  return (
    <main className="auth-shell"><section className="auth-card">
      <span className="auth-mark">STEM</span><h1>Trợ giảng STEM lớp 12</h1>
      <p>Chụp bài, nhận gợi ý từng bước và ôn đúng chủ đề mình đang yếu.</p>
      <form onSubmit={sendLink}><label htmlFor="email">Email</label>
        <input id="email" type="email" required value={email} onChange={e => setEmail(e.target.value)} placeholder="ban@example.com" />
        <button type="submit">Gửi liên kết đăng nhập</button></form>
      {status && <p className="auth-status">{status}</p>}
    </section></main>
  )
}

function ProgressPanel({ stages, problem, hintLevel, citations }) {
  const [draft, setDraft] = useState('')
  useEffect(() => { setDraft(problem?.problem_markdown || '') }, [problem])
  return (
    <aside className="progress-panel">
      <div className="panel-heading"><div><span className="eyebrow">Tiến trình học</span><h2>Phiên hiện tại</h2></div><span className="hint-pill">Gợi ý {hintLevel}/4</span></div>
      <ol className="stage-list">
        {stages.length === 0 && <li className="muted">Các bước xử lý sẽ xuất hiện tại đây.</li>}
        {stages.map((stage, index) => <li key={`${stage.stage}-${index}`} className={index === stages.length - 1 ? 'current' : 'complete'}><span>{index < stages.length - 1 ? '✓' : '•'}</span>{stage.label}</li>)}
      </ol>
      {problem && <section className="transcription-card"><span className="eyebrow">Bản chép đề · {Math.round(problem.confidence * 100)}%</span>
        <textarea value={draft} onChange={e => setDraft(e.target.value)} aria-label="Bản chép đề" />
        <p>Bạn có thể sửa rồi gửi nội dung này ở khung chat để xác nhận.</p></section>}
      {citations.length > 0 && <section className="source-list"><span className="eyebrow">Nguồn đã đối chiếu</span>
        {citations.map((item, index) => item.url?.startsWith('http')
          ? <a key={`${item.url}-${index}`} href={item.url} target="_blank" rel="noreferrer">{item.title}</a>
          : <span key={`${item.title}-${index}`}>{item.title} · học liệu riêng</span>)}</section>}
    </aside>
  )
}

function MasteryPanel({ mastery, summary, recommendations }) {
  const hasEvidence = (summary?.evidence_count || 0) > 0
  return (
    <aside className="mastery-panel"><div className="panel-heading"><div><span className="eyebrow">Đánh giá từ hội thoại</span><h2>Hồ sơ năng lực</h2></div></div>
      <section className={`profile-summary ${hasEvidence ? '' : 'empty'}`}>
        <div><strong>{summary?.level || 'Chưa đủ dữ liệu'}</strong>{summary?.overall_mastery != null && <b>{Math.round(summary.overall_mastery * 100)}%</b>}</div>
        <p>{summary?.narrative || 'Hồ sơ chỉ được tạo từ câu trả lời có thể đánh giá trong phiên học.'}</p>
        {hasEvidence && <small>{summary.evidence_count} câu trả lời · {summary.observed_skills} kỹ năng · độ tin cậy {Math.round(summary.confidence * 100)}%</small>}
        {summary?.strengths?.length > 0 && <div className="profile-signals"><span>Điểm mạnh</span>{summary.strengths.map(item => <em key={item}>{item}</em>)}</div>}
        {summary?.focus_areas?.length > 0 && <div className="profile-signals focus"><span>Cần củng cố</span>{summary.focus_areas.map(item => <em key={item}>{item}</em>)}</div>}
      </section>
      <div className="recommendation-list">{recommendations.length === 0 && <div className="empty-recommendations"><p className="muted">{hasEvidence ? 'Chưa có chủ đề ưu tiên mới.' : 'Hãy gửi bài và trình bày cách làm. Câu hỏi xin gợi ý sẽ không bị tính là làm sai.'}</p></div>}
        {recommendations.slice(0, 6).map(item => <article key={item.skill_id} className="recommendation-card"><div><span>{SUBJECT_LABEL[item.subject]}</span><b>{Math.round(item.score * 100)}</b></div><h3>{item.title}</h3><p>{item.reason}</p></article>)}</div>
      <div className="mastery-list">{mastery.map(item => <div className="mastery-row" key={item.skill_id}><span>{item.skill_id.split('.').slice(1).join(' ')}</span><div><i style={{ width: `${item.mastery * 100}%` }} /></div><b>{Math.round(item.mastery * 100)}%</b><small>{item.attempts} bằng chứng</small></div>)}</div>
    </aside>
  )
}

function KnowledgePanel({ subject, selectedDocumentIds, onSelectionChange, onClose }) {
  const [activeTab, setActiveTab] = useState('upload')
  const [file, setFile] = useState(null)
  const [title, setTitle] = useState('')
  const [busy, setBusy] = useState(false)
  const [jobs, setJobs] = useState([])
  const [documents, setDocuments] = useState([])
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true
    const refresh = () => Promise.all([
      apiFetch('/api/knowledge/jobs?limit=50').then(response => response.json()),
      apiFetch('/api/knowledge/documents?limit=50').then(response => response.json()),
    ])
      .then(([jobPayload, documentPayload]) => {
        if (active) {
          setJobs(jobPayload.items || [])
          setDocuments(documentPayload.items || [])
        }
      })
      .catch(err => { if (active) setError(err.message) })
    refresh()
    const timer = window.setInterval(refresh, 2000)
    return () => { active = false; window.clearInterval(timer) }
  }, [subject])

  const submit = async (event) => {
    event.preventDefault()
    if (!file || busy) return
    const formElement = event.currentTarget
    try {
      assertUploadSize(file)
    } catch (err) {
      setError(err.message)
      return
    }
    setBusy(true); setError('')
    try {
      const form = new FormData()
      form.append('file', file)
      form.append('subject', subject)
      if (title.trim()) form.append('title', title.trim())
      const response = await apiFetch('/api/knowledge', { method: 'POST', body: form })
      const queued = await response.json()
      setJobs(current => [{ ...queued, id: queued.job_id }, ...current.filter(item => item.id !== queued.job_id)])
      setFile(null); setTitle('')
      formElement.reset()
      setActiveTab('documents')
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  const documentIds = new Set(documents.map(document => document.id))
  const visibleJobs = jobs.filter(job => job.status !== 'completed' || !job.document_id || !documentIds.has(job.document_id))
  const counts = jobs.reduce((summary, job) => ({
    ...summary,
    [job.status]: (summary[job.status] || 0) + 1,
  }), { pending: 0, processing: 0, completed: 0, failed: 0 })
  const completedWithoutDocument = visibleJobs.filter(job => job.status === 'completed').length
  const completedCount = documents.length + completedWithoutDocument
  const libraryCount = counts.pending + counts.processing + counts.failed + completedCount

  return <div className="admin-overlay"><section className="admin-panel knowledge-panel">
    <header><div><span className="eyebrow">Kho RAG của bạn</span><h2>Thêm học liệu {SUBJECT_LABEL[subject]}</h2></div><button onClick={onClose}>Đóng</button></header>
    <nav className="knowledge-tabs" role="tablist" aria-label="Quản lý học liệu">
      <button role="tab" aria-selected={activeTab === 'upload'} className={activeTab === 'upload' ? 'active' : ''} onClick={() => setActiveTab('upload')}>Tải tài liệu</button>
      <button role="tab" aria-selected={activeTab === 'documents'} className={activeTab === 'documents' ? 'active' : ''} onClick={() => setActiveTab('documents')}>Tài liệu đã tải <span>{libraryCount}</span></button>
    </nav>
    {activeTab === 'upload' && <form onSubmit={submit}>
      <label>Tiêu đề <input value={title} onChange={event => setTitle(event.target.value)} placeholder="Tự lấy tên file nếu để trống" maxLength={300} /></label>
      <label>Tệp tài liệu <input required type="file" accept=".txt,.md,.markdown,.pdf,.docx" onChange={event => {
        const selected = event.target.files?.[0] || null
        try { assertUploadSize(selected); setFile(selected); setError('') } catch (err) { setFile(null); setError(err.message); event.target.value = '' }
      }} /></label>
      <small>Hỗ trợ TXT, Markdown, PDF, DOCX · tối đa 25 MB</small>
      <button type="submit" disabled={!file || busy}>{busy ? 'Đang tải lên…' : 'Tải lên và xử lý'}</button>
    </form>}
    {error && <div className="error-banner">{error}</div>}
    {activeTab === 'documents' && <section className="knowledge-library">
      <div className="knowledge-selection-note"><span>{selectedDocumentIds.length > 0 ? `${selectedDocumentIds.length} tài liệu được dùng làm nguồn cho chat` : 'Chưa chọn riêng: chat sẽ tìm trong toàn bộ kho học liệu'}</span>{selectedDocumentIds.length > 0 && <button onClick={() => onSelectionChange([])}>Dùng tất cả</button>}</div>
      <div className="knowledge-status-summary">
        <div><strong>{counts.pending}</strong><span>Đang chờ</span></div>
        <div><strong>{counts.processing}</strong><span>Đang xử lý</span></div>
        <div><strong>{completedCount}</strong><span>Hoàn tất</span></div>
        <div><strong>{counts.failed}</strong><span>Thất bại</span></div>
      </div>
      {libraryCount === 0 && <div className="knowledge-empty"><strong>Chưa có tài liệu</strong><span>Các tài liệu bạn tải lên sẽ xuất hiện tại đây.</span></div>}
      {libraryCount > 0 && <div className="knowledge-jobs">{visibleJobs.map(job => <article key={job.id || job.job_id} className={job.status}>
      <div><strong>{job.title}</strong><span>{job.status === 'pending' ? 'Đang chờ' : job.status === 'processing' ? 'Đang xử lý' : job.status === 'completed' ? 'Hoàn tất' : 'Thất bại'}</span></div>
      <small>{job.original_filename || job.filename} · {SUBJECT_LABEL[job.subject]}{job.status === 'completed' ? ` · ${job.chunks} đoạn · ${job.embedding_status === 'lexical_fallback' ? 'tìm kiếm từ khóa' : 'đã embedding'}` : ''}</small>
      {job.error && <p>{job.error}</p>}
      </article>)}{documents.map(document => <article key={document.id} className="completed">
        <div><strong>{document.title}</strong><span>Hoàn tất</span></div>
        <small>{document.original_filename || 'Tài liệu đã nhập'} · {SUBJECT_LABEL[document.subject]} · {document.chunks} đoạn · {document.embedding_status === 'lexical_fallback' ? 'tìm kiếm từ khóa' : 'đã embedding'}</small>
        <label className="knowledge-document-select"><input type="checkbox" checked={selectedDocumentIds.includes(document.id)} onChange={() => onSelectionChange(selectedDocumentIds.includes(document.id) ? selectedDocumentIds.filter(id => id !== document.id) : [...selectedDocumentIds, document.id])} /><span>Dùng tài liệu này trong chat</span></label>
      </article>)}</div>}
    </section>}
  </section></div>
}

function ChatHistoryPanel({ currentSessionId, onClose, onSelect }) {
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    apiFetch('/api/chat/sessions?limit=50')
      .then(response => response.json())
      .then(payload => setSessions(payload.items || []))
      .catch(err => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  return <div className="admin-overlay"><section className="admin-panel chat-history-panel">
    <header><div><span className="eyebrow">Dữ liệu hội thoại</span><h2>Lịch sử phiên chat</h2></div><button onClick={onClose}>Đóng</button></header>
    {loading && <p>Đang tải lịch sử…</p>}
    {error && <div className="error-banner">{error}</div>}
    {!loading && !error && sessions.length === 0 && <div className="history-empty"><strong>Chưa có phiên chat đã lưu</strong><span>Phiên sẽ xuất hiện sau khi bạn gửi tin nhắn đầu tiên và backend ghi nhận hội thoại.</span></div>}
    {sessions.length > 0 && <div className="chat-session-list">{sessions.map(session => <button key={session.session_id} className={session.session_id === currentSessionId ? 'active' : ''} onClick={() => onSelect(session)}>
      <div><strong>{session.title}</strong>{session.subject && <span>{SUBJECT_LABEL[session.subject]}</span>}</div>
      <p>{session.last_message}</p>
      <small>{session.message_count} tin nhắn · {new Date(session.updated_at).toLocaleString('vi-VN')}</small>
    </button>)}</div>}
  </section></div>
}

function ExamPlanPanel({ subject, onClose }) {
  const [plan, setPlan] = useState(null)
  const [error, setError] = useState('')
  const [targetScore, setTargetScore] = useState('')
  useEffect(() => {
    setPlan(null); setError('')
    const query = targetScore ? `?target_score=${encodeURIComponent(targetScore)}` : ''
    apiFetch(`/api/me/exam-plan/${subject}${query}`)
      .then(response => response.json())
      .then(setPlan)
      .catch(err => setError(err.message))
  }, [subject, targetScore])

  const research = plan?.research
  return <div className="admin-overlay"><section className="admin-panel exam-plan-panel">
    <header><div><span className="eyebrow">Research → Supervisor</span><h2>Lộ trình điểm thi {SUBJECT_LABEL[subject]}</h2></div><button onClick={onClose}>Đóng</button></header>
    <section className="exam-plan-guide">
      <div><h3>Cách sử dụng</h3><ol>
        <li>Chọn môn trên thanh điều hướng trước khi mở lộ trình.</li>
        <li>Chọn mức điểm mục tiêu để chỉ xem kế hoạch phù hợp, hoặc xem toàn bộ bốn thang điểm.</li>
        <li>Học theo thứ tự ưu tiên và các bước đề xuất. Tiếp tục làm bài trong chat để hệ thống cập nhật năng lực thực tế.</li>
      </ol></div>
      <label>Mục tiêu điểm
        <select value={targetScore} onChange={event => setTargetScore(event.target.value)}>
          <option value="">Xem tất cả thang điểm</option>
          <option value="5">Đạt 5 điểm</option>
          <option value="7">Đạt 7 điểm</option>
          <option value="8.5">Đạt 8,5 điểm</option>
          <option value="10">Hướng tới 10 điểm</option>
        </select>
      </label>
      <small>Lộ trình dựa trên câu thi đã duyệt của hai kỳ thi hoàn tất gần nhất và hồ sơ năng lực tích lũy từ hoạt động học của bạn.</small>
    </section>
    {!plan && !error && <p>Đang phân tích các câu hỏi đã duyệt của hai kỳ thi hoàn tất gần nhất…</p>}
    {error && <div className="error-banner">{error}</div>}
    {research && <>
      <section className={`research-summary ${research.status}`}>
        <div><strong>{research.status === 'ready' ? `Phân tích ${research.years.join('–')}` : research.status === 'partial' ? 'Dữ liệu một năm' : 'Chưa đủ dữ liệu'}</strong><span>{research.approved_question_count} câu đã duyệt</span></div>
        <small>Cập nhật {new Date(research.generated_at).toLocaleString('vi-VN')} · {research.timezone}</small>
        {research.notes.map((note, index) => <p key={index}>{note}</p>)}
      </section>
      {research.skills.length > 0 && <section className="exam-distribution"><h3>Phân bổ kiến thức</h3>{research.skills.slice(0, 8).map(skill => <div key={skill.skill_id}><span>{skill.title}</span><i><b style={{ width: `${Math.max(3, skill.weighted_prevalence * 100)}%` }} /></i><strong>{Math.round(skill.weighted_prevalence * 100)}%</strong><small>Độ khó {skill.average_difficulty}/5</small></div>)}</section>}
      {research.sources.length > 0 && <section className="exam-sources"><h3>Nguồn đề</h3>{research.sources.map(source => <a key={source.url} href={source.url} target="_blank" rel="noreferrer">{source.exam_year} · {source.publisher}</a>)}</section>}
      <div className="score-band-list">{plan.score_bands.map(item => <article key={item.band}>
        <header><strong>{item.band} điểm</strong><span>Mastery mục tiêu {Math.round(item.target_mastery * 100)}%</span></header>
        <small>Trọng tâm: {item.cognitive_focus.join(', ')} · cá nhân hóa {Math.round(item.personalization_confidence * 100)}%</small>
        <ol>{item.priorities.slice(0, 4).map(priority => <li key={priority.skill_id}><b>{priority.title}</b><span>{priority.reason}</span></li>)}</ol>
        <details><summary>Các bước thực hiện</summary><ul>{item.steps.map((step, index) => <li key={index}>{step}</li>)}</ul></details>
      </article>)}</div>
    </>}
  </section></div>
}

export default function App() {
  const [authSession, setAuthSession] = useState(supabase ? null : { user: { email: 'dev@local' } })
  const [loadingAuth, setLoadingAuth] = useState(Boolean(supabase))
  const [subject, setSubject] = useState('math')
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [asset, setAsset] = useState(null)
  const [busy, setBusy] = useState(false)
  const [hasStreamingToken, setHasStreamingToken] = useState(false)
  const [stages, setStages] = useState([])
  const [problem, setProblem] = useState(null)
  const [hintLevel, setHintLevel] = useState(0)
  const [citations, setCitations] = useState([])
  const [mastery, setMastery] = useState([])
  const [masterySummary, setMasterySummary] = useState(null)
  const [recommendations, setRecommendations] = useState([])
  const [selectedKnowledgeDocuments, setSelectedKnowledgeDocuments] = useState(() => {
    try {
      const stored = JSON.parse(localStorage.getItem(KNOWLEDGE_SELECTION_KEY) || '[]')
      return Array.isArray(stored) ? stored.filter(item => typeof item === 'string') : []
    } catch (_error) {
      return []
    }
  })
  const [error, setError] = useState('')
  const [showKnowledge, setShowKnowledge] = useState(false)
  const [showHistory, setShowHistory] = useState(false)
  const [showExamPlan, setShowExamPlan] = useState(false)
  const fileRef = useRef(null)
  const endRef = useRef(null)
  const currentSession = useRef(sessionId())

  const loadSession = useCallback(async (selectedSessionId) => {
    currentSession.current = selectedSessionId
    localStorage.setItem(SESSION_KEY, selectedSessionId)
    setStages([]); setCitations([]); setProblem(null); setHintLevel(0); setError('')
    const response = await apiFetch(`/api/history/${selectedSessionId}`)
    const payload = await response.json()
    setMessages(Array.isArray(payload.messages) ? payload.messages.map(item => ({ role: item.role, text: item.content, visualization: item.metadata?.visualization || null })) : [])
    if (payload.problem) setProblem(payload.problem)
    setHintLevel(payload.hint_level || 0)
  }, [])

  useEffect(() => {
    if (!supabase) return
    supabase.auth.getSession().then(({ data }) => { setAuthSession(data.session); setLoadingAuth(false) })
    const { data } = supabase.auth.onAuthStateChange((_event, nextSession) => setAuthSession(nextSession))
    return () => data.subscription.unsubscribe()
  }, [])

  const loadProfile = useCallback(async () => {
    if (!authSession) return
    const [masteryResponse, recResponse] = await Promise.all([apiFetch('/api/me/mastery'), apiFetch('/api/me/recommendations')])
    const masteryPayload = await masteryResponse.json()
    setMastery(masteryPayload.items)
    setMasterySummary(masteryPayload.summary)
    setRecommendations((await recResponse.json()).items)
  }, [authSession])
  useEffect(() => { loadProfile().catch(err => setError(err.message)) }, [loadProfile])
  useEffect(() => {
    localStorage.setItem(KNOWLEDGE_SELECTION_KEY, JSON.stringify(selectedKnowledgeDocuments))
  }, [selectedKnowledgeDocuments])
  useEffect(() => {
    if (!authSession) return
    loadSession(currentSession.current).catch(err => setError(err.message))
  }, [authSession, loadSession])
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, busy])

  const upload = async (file) => {
    if (!file) return
    setError('')
    assertUploadSize(file)
    const form = new FormData(); form.append('file', file)
    const response = await apiFetch('/api/uploads', { method: 'POST', body: form })
    const payload = await response.json()
    setAsset({ ...payload, name: file.name, localPreview: URL.createObjectURL(file) })
  }

  const handleEvent = (event) => {
    if (event.type === 'stage') setStages(current => [...current, event])
    if (event.type === 'problem_extracted') setProblem(event.problem)
    if (event.type === 'citation') setCitations(current => current.some(item => item.url === event.url) ? current : [...current, event])
    if (event.type === 'visualization' && event.success) {
      setHasStreamingToken(true)
      setMessages(current => {
        const last = current[current.length - 1]
        if (last?.role === 'assistant' && last.streaming) {
          return [...current.slice(0, -1), { ...last, visualization: event }]
        }
        return [...current, { role: 'assistant', text: '', visualization: event, streaming: true }]
      })
    }
    if (event.type === 'token') {
      setHasStreamingToken(true)
      setMessages(current => {
        const last = current[current.length - 1]
        if (last?.role === 'assistant' && last.streaming) {
          return [...current.slice(0, -1), { ...last, text: `${last.text}${event.content || ''}` }]
        }
        return [...current, { role: 'assistant', text: event.content || '', streaming: true }]
      })
    }
    if (event.type === 'hint' || event.type === 'feedback') {
      setHintLevel(event.hint_level || 0)
      setMessages(current => {
        const last = current[current.length - 1]
        if (last?.role === 'assistant' && last.streaming) {
          return [...current.slice(0, -1), { ...last, role: 'assistant', text: event.message, streaming: false }]
        }
        if (last?.role === 'assistant' && last.text === event.message) return current
        return [...current, { role: 'assistant', text: event.message }]
      })
      setHasStreamingToken(false)
    }
    if (event.type === 'mastery_updated') setMastery(current => [...current.filter(item => item.skill_id !== event.skill_id), { ...event, mastery: event.new_mastery }])
    if (event.type === 'recommendations_updated') setRecommendations(event.recommendations)
    if (event.type === 'error') setError(event.content)
    if (event.type === 'done') {
      setHasStreamingToken(false)
      setMessages(current => current.map(item => item.streaming ? { ...item, streaming: false } : item))
      loadProfile().catch(err => setError(err.message))
    }
  }

  const send = async () => {
    if (busy || (!input.trim() && !asset)) return
    const visibleText = input.trim() || 'Phân tích bài trong ảnh này.'
    const requestText = `Môn ${SUBJECT_LABEL[subject]}. ${visibleText}`
    setMessages(current => [...current, { role: 'user', text: visibleText, image: asset?.localPreview }])
    setInput(''); setBusy(true); setHasStreamingToken(false); setStages([]); setError('')
    const attachmentIds = asset ? [asset.asset_id] : []
    setAsset(null)
    try {
      const request = {
        session_id: currentSession.current,
        request_id: crypto.randomUUID(),
        message: requestText,
        display_message: visibleText,
        attachment_ids: attachmentIds,
        knowledge_document_ids: selectedKnowledgeDocuments,
        subject,
      }
      try {
        await streamTutorWebSocket(request, handleEvent)
      } catch (socketError) {
        if (!socketError.canFallback) throw socketError
        await streamTutorSse(request, handleEvent)
      }
    } catch (err) { setError(err.message) } finally { setBusy(false) }
  }

  const newSession = () => {
    currentSession.current = crypto.randomUUID(); localStorage.setItem(SESSION_KEY, currentSession.current)
    setMessages([]); setStages([]); setProblem(null); setCitations([]); setHintLevel(0); setHasStreamingToken(false)
    setShowHistory(false)
  }

  if (loadingAuth) return <div className="loading-screen">Đang khởi tạo…</div>
  if (!authSession) return <Login />

  return (
    <div className="app-shell"><header className="topbar"><div className="brand"><span>ST</span><div><b>STEM Tutor</b><small>CTGDPT 2018 · Lớp 12</small></div></div>
      <nav>{SUBJECTS.map(item => <button key={item.id} className={subject === item.id ? 'active' : ''} onClick={() => setSubject(item.id)}><i>{item.icon}</i>{item.label}</button>)}</nav>
      <div className="account-actions"><button onClick={() => setShowExamPlan(true)}>Lộ trình điểm</button><button onClick={() => setShowKnowledge(true)}>Học liệu</button><button onClick={() => setShowHistory(true)}>Lịch sử</button><button onClick={newSession}>Phiên mới</button>{supabase && <button onClick={() => supabase.auth.signOut()}>Đăng xuất</button>}</div></header>
      <div className="workspace"><ProgressPanel stages={stages} problem={problem} hintLevel={hintLevel} citations={citations} />
        <main className="chat-panel"><header><span className="subject-icon">{SUBJECTS.find(item => item.id === subject)?.icon}</span><div><h1>Học {SUBJECT_LABEL[subject]}</h1><p>Hãy gửi đề và cách bạn đã thử. Mình sẽ gợi mở từng bước.</p></div></header>
          <section className="messages">{messages.length === 0 && <div className="welcome"><span>✦</span><h2>Bắt đầu từ một bài bạn đang vướng</h2><p>Chụp rõ toàn bộ đề, hoặc nhập câu hỏi và phần bạn đã làm.</p></div>}
            {messages.map((message, index) => <article className={`message ${message.role}`} key={index}><span>{message.role === 'user' ? 'Bạn' : 'ST'}</span><div>{message.image && <img src={message.image} alt="Bài tập đã gửi" />}{message.visualization && <AuthenticatedVideo visualization={message.visualization} />}<Markdown>{message.text}</Markdown></div></article>)}
            {busy && !hasStreamingToken && <article className="message assistant"><span>ST</span><div className="typing"><i /><i /><i /></div></article>}<div ref={endRef} /></section>
          {error && <div className="error-banner">{error}<button onClick={() => setError('')}>×</button></div>}
          <footer className="composer">{asset && <div className="attachment"><img src={asset.localPreview} alt="preview" /><span>{asset.name}</span><button onClick={() => setAsset(null)}>×</button></div>}
            <div><button className="attach" onClick={() => fileRef.current?.click()} aria-label="Đính kèm ảnh">＋</button><input ref={fileRef} hidden type="file" accept="image/*" onChange={e => upload(e.target.files?.[0]).catch(err => setError(err.message))} />
              <textarea value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() } }} placeholder="Bạn đã thử bước nào?" />
              <button className="send" onClick={send} disabled={busy || (!input.trim() && !asset)}>Gửi</button></div><small>Ảnh tối đa 25 MB. Trợ giảng có thể mắc lỗi; hãy kiểm tra công thức và nguồn trước khi sử dụng.</small></footer>
        </main><MasteryPanel mastery={mastery} summary={masterySummary} recommendations={recommendations} /></div>
      {showKnowledge && <KnowledgePanel subject={subject} selectedDocumentIds={selectedKnowledgeDocuments} onSelectionChange={setSelectedKnowledgeDocuments} onClose={() => setShowKnowledge(false)} />}
      {showHistory && <ChatHistoryPanel currentSessionId={currentSession.current} onClose={() => setShowHistory(false)} onSelect={session => {
        if (session.subject) setSubject(session.subject)
        loadSession(session.session_id).then(() => setShowHistory(false)).catch(err => setError(err.message))
      }} />}
      {showExamPlan && <ExamPlanPanel subject={subject} onClose={() => setShowExamPlan(false)} />}
    </div>
  )
}
