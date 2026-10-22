import { useEffect, useRef, useState } from 'react'
import GeneralScene from './GeneralScene'

const number = value => Number(value.toPrecision(4)).toString()

export default function Illustration({ visualization }) {
  if (visualization.success && visualization.scene_data?.template === 'scene') return <GeneralScene scene={visualization.scene_data}/>
  return <LegacyIllustration visualization={visualization}/>
}

function LegacyIllustration({ visualization }) {
  const scene = visualization.scene_data
  const [playing, setPlaying] = useState(() => !window.matchMedia('(prefers-reduced-motion: reduce)').matches)
  const [time, setTime] = useState(0)
  const svgRef = useRef(null)
  const animated = ['simple_pendulum', 'harmonic_motion'].includes(scene?.template)
  useEffect(() => {
    if (!animated || !playing) return
    let frame, previous
    const tick = timestamp => {
      if (previous !== undefined) setTime(value => value + Math.min((timestamp - previous) / 1000, 0.05))
      previous = timestamp
      frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [animated, playing])
  useEffect(() => { setTime(0) }, [scene])
  if (!visualization.success || !scene) return <section className="visualization-card" role="status">
    <strong>{visualization.status === 'unsupported' ? 'Chưa có mẫu minh hoạ phù hợp' : 'Chưa tạo được minh hoạ'}</strong>
    <p>{visualization.error || 'Bạn vẫn có thể đọc lời giải bên dưới.'}</p>
  </section>

  const download = () => {
    const blob = new Blob([new XMLSerializer().serializeToString(svgRef.current)], { type: 'image/svg+xml' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url; link.download = `${scene.template}.svg`; link.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }
  return <section className="visualization-card illustration-card">
    <header><h3>{scene.title}</h3><span>{animated ? 'Mô phỏng tương tác' : 'Hình minh hoạ'}</span></header>
    <svg xmlns="http://www.w3.org/2000/svg" ref={svgRef} viewBox="0 0 640 380" role="img" aria-label={scene.title} style={{ width: '100%', background: '#f8fafc', color: '#0f172a', borderRadius: 12 }}>
      <title>{scene.title}</title>
      <g fontFamily="Arial, sans-serif" fontSize="14" fill="#0f172a">
        {animated ? <Physics scene={scene} time={time} /> : <Plot scene={scene} />}
      </g>
    </svg>
    <div className="illustration-controls">
      {animated && <><button type="button" onClick={() => setPlaying(value => !value)}>{playing ? 'Tạm dừng' : 'Chạy mô phỏng'}</button>
        <button type="button" onClick={() => setTime(0)}>Về ban đầu</button><span>t = {time.toFixed(1)} s</span></>}
      <button type="button" onClick={download}>Tải hình SVG</button>
    </div>
    <ul>{scene.assumptions?.map((note, index) => <li key={index}>{note}</li>)}</ul>
  </section>
}

function Plot({ scene }) {
  const sx = x => 65 + (x - scene.x_min) / (scene.x_max - scene.x_min) * 520
  const sy = y => 320 - (y - scene.y_min) / (scene.y_max - scene.y_min) * 265
  const tangent = scene.tangent
  // Unique clip IDs keep multiple illustrations on the same page independent.
  const id = useRef(`plot-${Math.random().toString(36).slice(2)}`).current
  return <>
    <defs><clipPath id={id}><rect x="65" y="55" width="520" height="265" /></clipPath></defs>
    {Array.from({ length: 6 }, (_, i) => {
      const x = scene.x_min + (scene.x_max - scene.x_min) * i / 5
      const y = scene.y_min + (scene.y_max - scene.y_min) * i / 5
      return <g key={i}><path d={`M${sx(x)},55V320 M65,${sy(y)}H585`} stroke="#dbe3ec" />
        <text x={sx(x)} y="342" textAnchor="middle">{number(x)}</text><text x="56" y={sy(y) + 4} textAnchor="end">{number(y)}</text></g>
    })}
    <g clipPath={`url(#${id})`} fill="none">
      {scene.x_min <= 0 && scene.x_max >= 0 && <path d={`M${sx(0)},55V320`} stroke="#64748b" />}
      {scene.y_min <= 0 && scene.y_max >= 0 && <path d={`M65,${sy(0)}H585`} stroke="#64748b" />}
      {scene.segments.map((segment, i) => <polyline key={i} points={segment.map(([x, y]) => `${sx(x)},${sy(y)}`).join(' ')} stroke="#2563eb" strokeWidth="2.5" />)}
      {tangent && <><line x1="65" y1={sy(tangent.y + tangent.slope * (scene.x_min - tangent.x))} x2="585" y2={sy(tangent.y + tangent.slope * (scene.x_max - tangent.x))} stroke="#dc2626" strokeWidth="2" />
        <circle cx={sx(tangent.x)} cy={sy(tangent.y)} r="5" fill="#dc2626" /></>}
    </g>
    <text x="65" y="26">y = {scene.expression}</text><text x="600" y="342">x</text>
    {tangent && <text x="65" y="370">Tiếp tuyến tại x = {number(tangent.x)}; hệ số góc ≈ {number(tangent.slope)}</text>}
  </>
}

function Physics({ scene, time }) {
  const pendulum = scene.template === 'simple_pendulum'
  const phase = scene.omega * time + (pendulum ? 0 : scene.phase)
  const theta = scene.angle_deg * Math.PI / 180 * Math.cos(phase)
  const displacement = Math.cos(phase)
  const bobX = pendulum ? 320 + 175 * Math.sin(theta) : 320 + 180 * displacement
  const bobY = pendulum ? 65 + 175 * Math.cos(theta) : 130
  const points = Array.from({ length: 201 }, (_, i) => {
    const t = i / 200 * 2 * scene.period
    const y = Math.cos(scene.omega * t + (pendulum ? 0 : scene.phase))
    return `${65 + i / 200 * 520},${305 - y * 35}`
  }).join(' ')
  const progress = (time % (2 * scene.period)) / (2 * scene.period)
  return <>
    <text x="30" y="26">T = {number(scene.period)} s; ω = {number(scene.omega)} rad/s</text>
    {pendulum ? <><text x="30" y="48">l = {scene.length_m} m; g = {scene.gravity} m/s²; θ₀ = {scene.angle_deg}°</text>
      <path d="M320,65V240" stroke="#94a3b8" strokeDasharray="5 4" />
      <line x1="320" y1="65" x2={bobX} y2={bobY} stroke="#334155" strokeWidth="3" /><circle cx="320" cy="65" r="5" fill="#334155" /></>
      : <><text x="30" y="48">A = {scene.amplitude}; φ = {scene.phase} rad; x = {number(scene.amplitude * displacement)}</text>
        <path d="M85,130H555 M320,95V165" stroke="#94a3b8" strokeDasharray="5 4" />
        <text x="85" y="175">−A</text><text x="315" y="175">0</text><text x="540" y="175">+A</text></>}
    <circle cx={bobX} cy={bobY} r="13" fill="#2563eb" />
    <text x="65" y="260">{pendulum ? 'θ/θ₀' : 'x/A'}</text>
    <path d="M65,305H585" stroke="#94a3b8" />
    <polyline points={points} stroke="#2563eb" fill="none" strokeWidth="2" />
    <circle cx={65 + progress * 520} cy={305 - displacement * 35} r="5" fill="#dc2626" />
    <text x="65" y="365">0</text><text x="320" y="365">T</text><text x="570" y="365">2T (s)</text>
  </>
}
