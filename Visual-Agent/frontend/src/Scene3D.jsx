import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { CSS2DObject, CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js'

const dispose = object => object.traverse(node => {
  node.geometry?.dispose()
  if(Array.isArray(node.material)) node.material.forEach(material=>material.dispose())
  else node.material?.dispose()
  if(node instanceof CSS2DObject) node.element.remove()
})

export default function Scene3D({scene, objects}) {
  const host = useRef(null), current = useRef(objects), rendererRef = useRef(null)
  const [error, setError] = useState('')
  current.current = objects
  useEffect(() => {
    const element = host.current
    let renderer, frame, resize, controls, world, labels
    try {
      renderer = new THREE.WebGLRenderer({antialias:true, preserveDrawingBuffer:true})
      rendererRef.current = renderer
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1,1.5))
      renderer.setClearColor('#f8fafc')
      element.appendChild(renderer.domElement)
      labels = new CSS2DRenderer(); labels.domElement.style.cssText='position:absolute;inset:0;pointer-events:none'
      element.appendChild(labels.domElement)
      world = new THREE.Scene()
      world.add(new THREE.AmbientLight(0xffffff,2))
      const light = new THREE.DirectionalLight(0xffffff,3);light.position.set(5,8,10);world.add(light)
      const bounds = new THREE.Box3()
      for(const object of [...scene.objects, ...scene.frames.flatMap(frame=>frame.objects)]) {
        const points=[...object.points,...(object.segments || []).flat()]
        const padding = ['sphere','point','circle','cylinder'].includes(object.kind) ? object.radius : object.kind==='box' ? Math.max(...object.size)/2 : 0
        for(const point of points) {
          bounds.expandByPoint(new THREE.Vector3(...point).addScalar(padding))
          bounds.expandByPoint(new THREE.Vector3(...point).addScalar(-padding))
        }
      }
      if(bounds.isEmpty())bounds.set(new THREE.Vector3(-1,-1,-1),new THREE.Vector3(1,1,1))
      const extent = Math.max(...bounds.getSize(new THREE.Vector3()).toArray(),scene.viewport[2]-scene.viewport[0], scene.viewport[3]-scene.viewport[1])
      const center = bounds.getCenter(new THREE.Vector3())
      const camera = new THREE.PerspectiveCamera(42,1,Math.max(0.001,extent/10000),extent*100)
      camera.position.copy(center).add(new THREE.Vector3(extent*0.9,extent*0.65,extent*1.2))
      controls = new OrbitControls(camera,renderer.domElement);controls.target.copy(center);controls.enableDamping=true;controls.update()
      const group = new THREE.Group();world.add(group)
      const size = () => {const width=element.clientWidth || 640; renderer.setSize(width,400);labels.setSize(width,400);camera.aspect=width/400;camera.updateProjectionMatrix()}
      resize = new ResizeObserver(size);resize.observe(element);size()
      let previous
      const draw = () => {
        if(previous !== current.current) {
          previous = current.current
          // Scene budgets are bounded; dispose geometry before rebuilding a changed frame.
          for(const child of [...group.children]) {dispose(child);group.remove(child)}
          for(const object of current.current) group.add(makeObject(object,extent,center))
        }
        controls.update();renderer.render(world,camera);labels.render(world,camera)
        frame=requestAnimationFrame(draw)
      }
      draw()
    } catch(exc) {setError(`Không thể hiển thị 3D: ${exc.message}`)}
    return () => {
      cancelAnimationFrame(frame);resize?.disconnect();controls?.dispose()
      if(world)dispose(world)
      renderer?.dispose();renderer?.forceContextLoss();element.replaceChildren();rendererRef.current=null
    }
  },[scene])
  const download = () => {
    const source=rendererRef.current.domElement, rect=source.getBoundingClientRect()
    const canvas=document.createElement('canvas');canvas.width=source.width;canvas.height=source.height
    const context=canvas.getContext('2d');context.drawImage(source,0,0)
    const ratio=canvas.width/rect.width
    // CSS2D labels are outside the WebGL framebuffer; include them in exports.
    for(const label of host.current.querySelectorAll('[data-scene-label]')) {
      const box=label.getBoundingClientRect()
      if(!box.width||!box.height)continue
      const x=(box.left-rect.left)*ratio,y=(box.top-rect.top)*ratio
      const style=getComputedStyle(label)
      context.fillStyle='rgba(248,250,252,.9)';context.fillRect(x,y,box.width*ratio,box.height*ratio)
      context.fillStyle=style.color;context.font=`${14*ratio}px Arial`;context.textBaseline='top'
      const maxWidth=(box.width-8)*ratio, lines=[]
      for(const paragraph of label.textContent.split('\n')) {
        let line=''
        for(const word of paragraph.split(/\s+/)) {
          const next=line?`${line} ${word}`:word
          if(line&&context.measureText(next).width>maxWidth){lines.push(line);line=word}else line=next
        }
        lines.push(line)
      }
      lines.forEach((line,index)=>context.fillText(line,x+4*ratio,y+(2+index*16)*ratio))
    }
    const link=document.createElement('a');link.href=canvas.toDataURL('image/png');link.download='illustration-3d.png';link.click()
  }
  return <><div ref={host} role="img" aria-label={scene.title} style={{position:'relative',height:400,width:'100%',overflow:'hidden'}}/>
    {error ? <p role="status">{error}</p> : <button type="button" onClick={download}>Tải ảnh PNG</button>}</>
}

function makeObject(object, extent, center) {
  const group=new THREE.Group()
  const vector = point => new THREE.Vector3(...point)
  const points=object.points.map(vector)
  const color=new THREE.Color(object.color)
  const material=()=>new THREE.MeshStandardMaterial({color,opacity:object.opacity,transparent:object.opacity<1,side:THREE.DoubleSide})
  const line = values => {
    const geometry=new THREE.BufferGeometry().setFromPoints(values)
    return new THREE.Line(geometry,new THREE.LineBasicMaterial({color,opacity:object.opacity,transparent:object.opacity<1}))
  }
  const kind=object.kind
  if(kind==='point'||kind==='sphere') {
    const mesh=new THREE.Mesh(new THREE.SphereGeometry(object.radius,20,12),material());mesh.position.copy(points[0]);group.add(mesh)
  } else if(kind==='box') {
    const mesh=new THREE.Mesh(new THREE.BoxGeometry(...object.size),material());mesh.material.wireframe=!object.fill;mesh.position.copy(points[0]);group.add(mesh)
  } else if(kind==='cylinder') {
    const direction=points[1].clone().sub(points[0]),length=direction.length()
    const mesh=new THREE.Mesh(new THREE.CylinderGeometry(object.radius,object.radius,length,16),material())
    mesh.position.copy(points[0]).add(points[1]).multiplyScalar(0.5);mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),direction.normalize());group.add(mesh)
  } else if(kind==='arrow') {
    const direction=points[1].clone().sub(points[0]), length=direction.length()
    if(length>=1e-8)group.add(new THREE.ArrowHelper(direction.normalize(),points[0],length,color,Math.min(length*0.2,extent*0.03),Math.min(length*0.1,extent*0.015)))
  } else if(kind==='axes') group.add(new THREE.AxesHelper(extent/2))
  else if(kind==='curve') for(const segment of object.segments)group.add(line(segment.map(vector)))
  else if(kind==='polygon') {
    group.add(line([...points,points[0]]))
    if(object.fill) {
      const normal=points[1].clone().sub(points[0]).cross(points[2].clone().sub(points[0]))
      const components=[Math.abs(normal.x),Math.abs(normal.y),Math.abs(normal.z)]
      const axis=components.indexOf(Math.max(...components))
      const contour=points.map(point=>axis===0 ? new THREE.Vector2(point.y,point.z) : axis===1 ? new THREE.Vector2(point.x,point.z) : new THREE.Vector2(point.x,point.y))
      const geometry=new THREE.BufferGeometry().setFromPoints(points)
      geometry.setIndex(THREE.ShapeUtils.triangulateShape(contour,[]).flat());geometry.computeVertexNormals()
      group.add(new THREE.Mesh(geometry,material()))
    }
  } else if(kind==='circle'||kind==='ellipse') {
    const rx=kind==='circle'?object.radius:object.size[0], ry=kind==='circle'?object.radius:object.size[1]
    const values=Array.from({length:65},(_,i)=>points[0].clone().add(new THREE.Vector3(rx*Math.cos(i*Math.PI/32),ry*Math.sin(i*Math.PI/32),0)))
    group.add(line(values))
    if(object.fill){const mesh=new THREE.Mesh(new THREE.CircleGeometry(1,64),material());mesh.scale.set(rx,ry,1);mesh.position.copy(points[0]);group.add(mesh)}
  } else if(kind!=='text') {
    if(kind==='line') {
      const direction=points[1].clone().sub(points[0]).normalize()
      const anchor=points[0].clone().addScaledVector(direction,center.clone().sub(points[0]).dot(direction))
      group.add(line([anchor.clone().addScaledVector(direction,-extent*3),anchor.clone().addScaledVector(direction,extent*3)]))
    } else group.add(line(points))
  }
  if(object.label && points.length) {
    const div=document.createElement('div');div.textContent=object.label
    div.dataset.sceneLabel=object.id
    div.style.cssText=`font:14px Arial,sans-serif;color:${object.color};background:rgba(248,250,252,.85);padding:2px 4px;white-space:pre-wrap;max-width:220px`
    const label=new CSS2DObject(div);label.position.copy(points[0]);if(kind!=='text')label.position.y+=object.radius+extent*0.02;group.add(label)
  }
  group.traverse(node=>{if(node.material){node.material.opacity=object.opacity;node.material.transparent=object.opacity<1}})
  return group
}
