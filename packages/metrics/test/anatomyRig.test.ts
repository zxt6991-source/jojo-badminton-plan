import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {createAnatomyRig} from '../../../apps/web/src/anatomyRig.ts';
import {createPose3DSequence,type Pose3DFrame} from '../../../apps/web/src/pose3d.ts';

async function fixture(){
  const [model,video,points]=await Promise.all([
    readFile(new URL('../../../apps/web/public/models/anatomy/body.glb',import.meta.url)),
    readFile(new URL('../../../fixtures/demo-video.json',import.meta.url),'utf8'),
    readFile(new URL('../../../fixtures/keypoints.f32',import.meta.url)),
  ]);
  const scene=(await new GLTFLoader().parseAsync(model.buffer.slice(model.byteOffset,model.byteOffset+model.byteLength),'')).scene;
  const sequence=createPose3DSequence(JSON.parse(video),new Float32Array(points.buffer.slice(points.byteOffset,points.byteOffset+points.byteLength)));
  return{scene,sequence,rig:createAnatomyRig(scene)};
}
const vertex=(mesh:THREE.SkinnedMesh,index:number)=>mesh.applyBoneTransform(index,new THREE.Vector3().fromBufferAttribute(mesh.geometry.getAttribute('position'),index));
const near=(a:number,b:number)=>assert.ok(Math.abs(a-b)<1e-6,`${a} != ${b}`);

test('anatomical meshes themselves articulate and preserve rigid bone shape',async()=>{
  const {rig,sequence}=await fixture();
  try{
    const humerus=rig.meshes.find(mesh=>mesh.userData.contextType==='bone'&&mesh.userData.rigBone==='upperArmR')!;
    const muscle=rig.meshes.find(mesh=>mesh.userData.muscleId==='biceps_brachii_long_right')!;
    assert.ok(humerus.isSkinnedMesh&&muscle.isSkinnedMesh);
    assert.equal(humerus.skeleton,muscle.skeleton);
    rig.update(sequence.frame(116));const initial=vertex(humerus,0),distance=initial.distanceTo(vertex(humerus,30)),muscleInitial=vertex(muscle,0);
    rig.update(sequence.frame(160));
    assert.ok(initial.distanceTo(vertex(humerus,0))>.1,'bone vertices must move with swing');
    assert.ok(muscleInitial.distanceTo(vertex(muscle,0))>.1,'actual muscle surface must move too');
    near(distance,vertex(humerus,0).distanceTo(vertex(humerus,30)));
    for(const mesh of rig.meshes)for(let i=0;i<mesh.geometry.getAttribute('position').count;i+=31)assert.ok(vertex(mesh,i).toArray().every(Number.isFinite));
  }finally{rig.dispose();}
});

test('retargeting retains fixed anatomical segment lengths across the clip',async()=>{
  const {rig,sequence}=await fixture();
  try{
    const pairs=[[5,7],[7,9],[6,8],[8,10],[11,13],[13,15],[12,14],[14,16]];
    const reference=rig.sample(sequence.frame(116));
    for(const frame of [125,140,160,180,250]){
      const joints=rig.sample(sequence.frame(frame));
      for(const [a,b] of pairs)near(reference[a]!.distanceTo(reference[b]!),joints[a]!.distanceTo(joints[b]!));
    }
  }finally{rig.dispose();}
});

test('trajectory sampling cannot mutate the displayed pose or source mesh',async()=>{
  const {rig,sequence,scene}=await fixture();
  try{
    let original:THREE.Mesh|undefined;scene.traverse(object=>{if(object instanceof THREE.Mesh&&!original)original=object;});
    const sourcePosition=original!.geometry.getAttribute('position').clone();
    assert.ok(rig.meshes.every(mesh=>mesh.geometry!==original!.geometry));
    rig.update(sequence.frame(160));const wrist=rig.jointPositions[10]!.clone(),mesh=rig.meshes[0],displayVertex=vertex(mesh,0);
    const previous=rig.sample(sequence.frame(116));assert.ok(previous[10]!.distanceTo(wrist)>.1);
    near(wrist.distanceTo(rig.jointPositions[10]!),0);near(displayVertex.distanceTo(vertex(mesh,0)),0);
    assert.deepEqual(original!.geometry.getAttribute('position').array,sourcePosition.array);
  }finally{rig.dispose();}
});

test('missing joints hide attached anatomical surfaces and recover without old poses',async()=>{
  const {rig,sequence}=await fixture();
  try{
    rig.update(sequence.frame(160));assert.ok(rig.jointPositions[10]);
    rig.update(sequence.frame(596));assert.equal(rig.jointPositions[10],null);
    const rightForearm=rig.meshes.find(mesh=>mesh.userData.rigBone==='forearmR')!;
    assert.equal(rightForearm.userData.poseVisible,false);assert.equal(rightForearm.visible,false);
    assert.ok(rig.missingParts>0);
    const missing=sequence.frame(596);for(const joint of missing.joints){joint.visible=false;joint.position=[NaN,NaN,NaN];}
    rig.update(missing);assert.ok(rig.meshes.every(mesh=>mesh.userData.poseVisible===false));
    rig.update(sequence.frame(116));assert.ok(rig.jointPositions[10]);assert.equal(rightForearm.userData.poseVisible,true);
    assert.ok(rig.meshes.every(mesh=>mesh.skeleton.bones.every(bone=>bone.matrixWorld.elements.every(Number.isFinite))));
  }finally{rig.dispose();}
});

// A deliberately planar observation. Anatomical left is screen-right when
// facing the camera, and screen-left when facing away. Z is display decoration.
function planarPose(back=false):Pose3DFrame {
  const observed:Record<number,[number,number]>={
    0:[0,.77],1:[.045,.80],2:[-.045,.80],3:[.08,.78],4:[-.08,.78],
    5:[.20,.56],6:[-.20,.56],7:[.25,.27],8:[-.25,.27],9:[.29,.02],10:[-.29,.02],
    11:[.09,0],12:[-.09,0],13:[.10,-.44],14:[-.10,-.44],15:[.10,-.81],16:[-.10,-.81],
    17:[.12,-.85],20:[-.12,-.85],
  };
  const joints=Array.from({length:65},(_,id)=>({
    position:observed[id]?[observed[id][0]*(back?-1:1),observed[id][1],id%2?.1:-.1] as [number,number,number]:[NaN,NaN,NaN] as [number,number,number],
    confidence:observed[id]?.95:0,visible:!!observed[id],
  }));
  return{joints,mode:'schematic-2d',origin:[0,0,0],scale:1,visibleCount:Object.keys(observed).length};
}
const bone=(rig:ReturnType<typeof createAnatomyRig>,name:string)=>rig.root.getObjectByName(name) as THREE.Bone;
const forward=(rig:ReturnType<typeof createAnatomyRig>,name:string)=>new THREE.Vector3(0,0,1).applyQuaternion(bone(rig,name).quaternion);

test('nose mislocalization cannot reverse or pitch the head away from torso facing',()=>{
  const rig=createAnatomyRig(new THREE.Group());
  try{
    for(const back of [false,true]){
      const pose=planarPose(back);
      // A single detected nose is insufficient evidence for 3D gaze.
      for(const id of [1,2,3,4])pose.joints[id].visible=false;
      for(const [x,y] of [[0,.77],[.16,.72],[-.16,.92]]){
        pose.joints[0].position=[x,y,0];rig.update(pose);
        assert.ok(rig.jointPositions[0],'plausible nose retains a neutral head');
        near(forward(rig,'head').dot(forward(rig,'torso')),1);
        near(forward(rig,'head').z,back?-1:1);
        near(bone(rig,'head').quaternion.angleTo(bone(rig,'torso').quaternion),0);
      }
    }
  }finally{rig.dispose();}
});

test('paired face observations only permit bounded head roll without changing facing',()=>{
  const rig=createAnatomyRig(new THREE.Group());
  try{
    for(const back of [false,true]){
      const pose=planarPose(back);
      pose.joints[3].position[1]=.88;pose.joints[4].position[1]=.73;
      rig.update(pose);
      const tilt=bone(rig,'head').quaternion.angleTo(bone(rig,'torso').quaternion);
      assert.ok(tilt>.1&&tilt<Math.PI/8,'head tilt must be visible but conservatively limited');
      near(forward(rig,'head').dot(forward(rig,'torso')),1);
      // Swapped facial labels must not turn a head through 180 degrees.
      for(const [a,b] of [[1,2],[3,4]])[pose.joints[a],pose.joints[b]]=[pose.joints[b],pose.joints[a]];
      rig.update(pose);near(bone(rig,'head').quaternion.angleTo(bone(rig,'torso').quaternion),0);
    }
  }finally{rig.dispose();}
});

test('schematic Z offsets cannot change anatomical rotations or retargeted landmarks',()=>{
  const rig=createAnatomyRig(new THREE.Group());
  try{
    const pose=planarPose(),source=structuredClone(pose);
    rig.update(pose);
    const rotations=rig.root.children.map(object=>object.quaternion.clone()),joints=rig.sample(pose);
    const depthChanged=structuredClone(pose);
    for(let id=0;id<depthChanged.joints.length;id++)if(depthChanged.joints[id].visible)depthChanged.joints[id].position[2]=(id%3-1)*2;
    rig.update(depthChanged);
    rig.root.children.forEach((object,i)=>near(rotations[i].angleTo(object.quaternion),0));
    assert.deepEqual(rig.sample(depthChanged),joints);assert.deepEqual(pose,source);
  }finally{rig.dispose();}
});

test('missing or implausible face detections hide head surfaces and recover on the next usable pose',async()=>{
  const {rig}=await fixture();
  try{
    const pose=planarPose(),head=rig.meshes.find(mesh=>mesh.userData.contextType==='bone'&&mesh.userData.rigBone==='head')!;
    const torso=rig.meshes.find(mesh=>mesh.userData.contextType==='bone'&&mesh.userData.rigBone==='torso')!;
    rig.update(pose);assert.equal(head.visible,true);
    for(const condition of ['missing','low-confidence','outlier']){
      const bad=structuredClone(pose);
      for(let id=0;id<5;id++){
        if(condition==='missing')bad.joints[id].visible=false;
        if(condition==='low-confidence')bad.joints[id].confidence=.49;
        if(condition==='outlier')bad.joints[id].position=[2,-2,0];
      }
      rig.update(bad);assert.equal(rig.jointPositions[0],null);assert.equal(head.visible,false);assert.equal(torso.visible,true);
      rig.update(pose);assert.equal(head.visible,true);assert.ok(rig.jointPositions[0]);
    }
  }finally{rig.dispose();}
});

test('bad endpoint detections cannot fold elbows and knees closed or reverse feet',()=>{
  const rig=createAnatomyRig(new THREE.Group());
  try{
    const pose=planarPose();
    // Return both wrists almost to the shoulders and both ankles to the hips.
    for(const [w,s] of [[9,5],[10,6],[15,11],[16,12]])pose.joints[w].position=[...pose.joints[s].position];
    pose.joints[17].position=[-.9,.9,0];pose.joints[20].position=[.9,.9,0];
    rig.update(pose);const joints=rig.jointPositions;
    for(const [root,hinge,end,minOpening] of [[5,7,9,20],[6,8,10,20],[11,13,15,25],[12,14,16,25]]){
      assert.ok(joints[root]&&joints[hinge]&&joints[end]);
      const opening=joints[root]!.clone().sub(joints[hinge]!).angleTo(joints[end]!.clone().sub(joints[hinge]!));
      assert.ok(opening>THREE.MathUtils.degToRad(minOpening),`joint ${hinge} must not collapse on itself`);
    }
    for(const side of ['L','R']){
      const relative=bone(rig,`foot${side}`).quaternion.angleTo(bone(rig,`shin${side}`).quaternion);
      assert.ok(relative<Math.PI/4,'toe outliers must not rotate feet independently through a half turn');
    }
  }finally{rig.dispose();}
});

test('random-access pose sampling and narrow torso fallback do not retain another frame',()=>{
  const rig=createAnatomyRig(new THREE.Group());
  try{
    const front=planarPose(),back=planarPose(true),narrow=planarPose();
    narrow.joints[5].position[0]=.005;narrow.joints[6].position[0]=-.005;
    const expected=rig.sample(front),before=structuredClone(front);
    rig.update(back);const displayed=rig.root.children.map(object=>object.matrixWorld.clone());
    for(const pose of [narrow,front,back,front,narrow])rig.sample(pose);
    assert.deepEqual(rig.sample(front),expected);assert.deepEqual(front,before);
    rig.root.children.forEach((object,i)=>assert.deepEqual(object.matrixWorld,displayed[i]));
    rig.update(narrow);near(forward(rig,'torso').z,1);assert.ok(rig.jointPositions[0]);
    narrow.joints[11].position[0]=.002;narrow.joints[12].position[0]=-.002;
    rig.update(narrow);assert.ok(Object.values(rig.jointPositions).every(point=>point===null));
    rig.update(front);assert.deepEqual(rig.jointPositions,expected);
  }finally{rig.dispose();}
});

test('trapezius skull attachments follow head roll while lower back vertices stay on the torso',async()=>{
  const {rig}=await fixture();
  try{
    const muscles=rig.meshes.filter(mesh=>String(mesh.userData.muscleId).startsWith('trapezius_lower_'));
    assert.equal(muscles.length,2,'exercise both actual atlas trapezius meshes');
    const pose=planarPose();rig.update(pose);
    const neutralHeadInverse=bone(rig,'head').matrixWorld.clone().invert();
    const samples=muscles.flatMap(mesh=>{
      const position=mesh.geometry.getAttribute('position');
      return Array.from({length:position.count},(_,index)=>({mesh,index,y:position.getY(index),before:vertex(mesh,index)}))
        .filter(sample=>sample.y>1.54||sample.y<1.44);
    });
    assert.ok(samples.some(sample=>sample.y>1.54),'atlas includes the high skull attachments');
    assert.ok(samples.some(sample=>sample.y<1.44),'atlas includes the lower torso attachments');
    pose.joints[3].position[1]=.88;pose.joints[4].position[1]=.73;
    rig.update(pose);
    const headChange=bone(rig,'head').matrixWorld.clone().multiply(neutralHeadInverse);
    let movedAttachments=0;
    for(const sample of samples){
      const after=vertex(sample.mesh,sample.index);
      if(sample.y>1.54){
        near(after.distanceTo(sample.before.clone().applyMatrix4(headChange)),0);
        if(after.distanceTo(sample.before)>.002)movedAttachments++;
      }else near(after.distanceTo(sample.before),0);
    }
    assert.ok(movedAttachments>0,'head roll must move the upper muscle attachments, not leave them on the torso');
  }finally{rig.dispose();}
});
