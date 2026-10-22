import { test } from 'node:test'
import assert from 'node:assert/strict'
import { objectsAtTime, lineEndpoints } from '../src/sceneState.mjs'

test('interpolates only animated numeric objects and keeps static objects',()=>{
 const staticObject={id:'axis',points:[[0,0,0],[1,0,0]]}
 const a={id:'ball',points:[[0,0,0]],radius:1,size:[1,1,1]}
 const b={...a,points:[[4,2,0]]}
 const result=objectsAtTime({objects:[staticObject,a],frames:[{time:0,objects:[a]},{time:2,objects:[b]}]},1)
 assert.equal(result[0],staticObject)
 assert.deepEqual(result[1].points,[[2,1,0]])
})

test('clips an infinite line to the viewport without changing geometry',()=>{
 assert.deepEqual(lineEndpoints([[0,0],[1,1]],[-2,-1,3,2]),[[-1,-1],[2,2]])
 assert.deepEqual(lineEndpoints([[5,0],[5,1]],[-1,-1,1,1]),[])
})
