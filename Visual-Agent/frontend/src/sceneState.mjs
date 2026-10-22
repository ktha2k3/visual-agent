// Only interpolate already validated numeric frames; no formula evaluation in the browser.
export function objectsAtTime(scene, time) {
  if (!scene.frames?.length) return scene.objects
  const frames = scene.frames
  let lo = 0, hi = frames.length - 1
  while (lo + 1 < hi) {
    const mid = (lo + hi) >> 1
    if (frames[mid].time <= time) lo = mid
    else hi = mid
  }
  const first = frames[lo], second = frames[hi]
  const fraction = Math.max(0, Math.min(1, (time - first.time) / (second.time - first.time || 1)))
  const interpolate = (a, b) => {
    if (typeof a === 'number' && typeof b === 'number') return a + (b-a) * fraction
    if (Array.isArray(a) && Array.isArray(b) && a.length === b.length) return a.map((value, i) => interpolate(value, b[i]))
    return fraction < 0.5 ? a : b
  }
  const updates = new Map(first.objects.map((object, index) => {
    const next = second.objects[index]
    return [object.id, { ...object, points: interpolate(object.points, next.points),
      radius: interpolate(object.radius, next.radius), size: interpolate(object.size, next.size),
      segments: interpolate(object.segments, next.segments) }]
  }))
  return scene.objects.map(object => updates.get(object.id) || object)
}

export function lineEndpoints(points, viewport) {
  const [a, b] = points
  const dx = b[0]-a[0], dy = b[1]-a[1]
  let min = -Infinity, max = Infinity
  for (const [origin, direction, start, end] of [[a[0], dx, viewport[0], viewport[2]], [a[1], dy, viewport[1], viewport[3]]]) {
    if (Math.abs(direction) < 1e-12) { if (origin < start || origin > end) return []; continue }
    const first = (start-origin)/direction, second = (end-origin)/direction
    min = Math.max(min, Math.min(first, second)); max = Math.min(max, Math.max(first, second))
  }
  if (min > max || !Number.isFinite(min) || !Number.isFinite(max)) return []
  return [[a[0]+min*dx, a[1]+min*dy], [a[0]+max*dx, a[1]+max*dy]]
}
