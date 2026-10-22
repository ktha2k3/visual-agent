import { lazy, Suspense, useEffect, useId, useMemo, useRef, useState } from 'react'
import { lineEndpoints, objectsAtTime } from './sceneState.mjs'

const Scene3D = lazy(() => import('./Scene3D.jsx'))

export default function GeneralScene({ scene }) {
  const [time, setTime] = useState(0)
  const [speed, setSpeed] = useState(1)
  const [playing, setPlaying] = useState(() => !window.matchMedia('(prefers-reduced-motion: reduce)').matches)
  const svg = useRef(null)
  const clip = useId().replaceAll(':', '')
  const objects = useMemo(() => objectsAtTime(scene, time), [scene, time])
  useEffect(() => { setTime(0) }, [scene])
  useEffect(() => {
    if (!playing || !scene.duration) return
    let frame, previous, elapsed = time
    const tick = now => {
      if (previous !== undefined) elapsed = Math.min(scene.duration, elapsed + Math.min((now-previous)/1000, 0.1)*speed)
      previous = now; setTime(elapsed)
      if (elapsed < scene.duration) frame = requestAnimationFrame(tick)
      else setPlaying(false)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
    // time is captured when playback starts; scrubbing pauses playback.
  }, [playing, speed, scene])
  const [xmin, ymin, xmax, ymax] = scene.viewport
  const scale = Math.min(550/(xmax-xmin), 320/(ymax-ymin))
  const sx = x => 320 + (x - (xmin+xmax)/2)*scale
  const sy = y => 200 - (y - (ymin+ymax)/2)*scale
  const pointString = points => points.map(point => `${sx(point[0])},${sy(point[1])}`).join(' ')
  const saveSVG = () => {
    const url = URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(svg.current)], {type:'image/svg+xml'}))
    const link = document.createElement('a'); link.href = url; link.download = 'illustration.svg'; link.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  }
  return <section className="visualization-card illustration-card">
    <header><h3>{scene.title}</h3><span>{scene.dimension === 3 ? 'Cảnh 3D · kéo để xoay' : 'Minh hoạ theo đề bài'}</span></header>
    {scene.dimension === 3 ? <Suspense fallback={<p>Đang tải bộ hiển thị 3D…</p>}><Scene3D scene={scene} objects={objects}/></Suspense>
      : <svg ref={svg} xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 400" role="img" aria-label={scene.title} style={{width:'100%', background:'#f8fafc', borderRadius:12}}>
        <title>{scene.title}</title>
        <defs><clipPath id={clip}><rect x={sx(xmin)} y={sy(ymax)} width={(xmax-xmin)*scale} height={(ymax-ymin)*scale}/></clipPath></defs>
        <g clipPath={`url(#${clip})`}>
          {objects.map(object => {
            const {id, kind, points, color, opacity, width, fill, radius, size, label} = object
            const anchor = points[0]
            const style = {stroke:color, strokeWidth:width, fill:fill ? color : 'none', strokeLinejoin:'round'}
            let shape
            if (kind === 'point' || kind === 'sphere') shape = <circle cx={sx(anchor[0])} cy={sy(anchor[1])} r={Math.max(3,radius*scale)} fill={color}/>
            else if (kind === 'circle' || kind === 'ellipse') shape = <ellipse cx={sx(anchor[0])} cy={sy(anchor[1])} rx={(kind === 'circle' ? radius : size[0])*scale} ry={(kind === 'circle' ? radius : size[1])*scale} {...style}/>
            else if (kind === 'polygon') shape = <polygon points={pointString(points)} {...style}/>
            else if (kind === 'box') shape = <rect x={sx(anchor[0]-size[0]/2)} y={sy(anchor[1]+size[1]/2)} width={size[0]*scale} height={size[1]*scale} {...style}/>
            else if (kind === 'axes') shape = <g stroke={color} strokeWidth={width}><path d={`M${sx(xmin)},${sy(0)}H${sx(xmax)} M${sx(0)},${sy(ymin)}V${sy(ymax)}`} fill="none"/></g>
            else if (kind === 'curve') shape = object.segments.map((segment,i) => <polyline key={i} points={pointString(segment)} {...style} fill="none"/> )
            else if (kind !== 'text') {
              const coords = kind === 'line' ? lineEndpoints(points, scene.viewport) : points
              shape = coords.length > 1 && <g><polyline points={pointString(coords)} {...style} fill="none"/>
                {kind === 'arrow' && (() => {
                  const a=coords[coords.length-2], b=coords[coords.length-1]
                  const distance=Math.hypot(sx(b[0])-sx(a[0]),sy(b[1])-sy(a[1]))
                  if(distance<1e-8)return null
                  const angle=Math.atan2(sy(b[1])-sy(a[1]),sx(b[0])-sx(a[0]))
                  const x=sx(b[0]), y=sy(b[1]), length=Math.min(10,distance*0.4)
                  return <polygon points={`${x},${y} ${x-length*Math.cos(angle-0.4)},${y-length*Math.sin(angle-0.4)} ${x-length*Math.cos(angle+0.4)},${y-length*Math.sin(angle+0.4)}`} fill={color}/>
                })()}</g>
            }
            return <g key={id} opacity={opacity} data-object-id={id}>{shape}
              {label && anchor && <text x={sx(anchor[0]) + (kind === 'text' ? 0 : 8)} y={sy(anchor[1]) - (kind === 'text' ? 0 : 8)} fill={color} fontSize="14" fontFamily="Arial, sans-serif">{label}</text>}
            </g>
          })}
        </g>
      </svg>}
    <div className="illustration-controls">
      {scene.duration > 0 && <><button type="button" onClick={() => {if(time>=scene.duration)setTime(0); setPlaying(value=>!value)}}>{playing ? 'Tạm dừng' : 'Chạy mô phỏng'}</button>
        <button type="button" onClick={() => {setPlaying(false);setTime(0)}}>Về ban đầu</button>
        <label>Thời gian <input aria-label="Thời gian mô phỏng" type="range" min="0" max={scene.duration} step={scene.duration/300} value={time} onChange={event=>{setPlaying(false);setTime(Number(event.target.value))}}/></label>
        <span>{time.toFixed(2)} / {scene.duration} s</span>
        <label>Tốc độ <select aria-label="Tốc độ mô phỏng" value={speed} onChange={event=>setSpeed(Number(event.target.value))}>{[0.25,0.5,1,2].map(value=><option key={value} value={value}>{value}×</option>)}</select></label></>}
      {scene.dimension === 2 && <button type="button" onClick={saveSVG}>Tải hình SVG</button>}
    </div>
    {scene.parameters?.length > 0 && <p>{scene.parameters.map(param => `${param.name} = ${param.value}${param.unit ? ` ${param.unit}` : ''}`).join('; ')}</p>}
    {scene.assumptions?.length > 0 && <ul>{scene.assumptions.map((note,index)=><li key={index}>{note}</li>)}</ul>}
  </section>
}
