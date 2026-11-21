// Anatomical rig definitions and skin adaptation: CC-BY-SA-4.0.
// See public/models/anatomy/MOTION-NOTICE.txt for model provenance.
import * as THREE from 'three';
import type {Pose3DFrame} from './pose3d';

type JointMap=Record<number,THREE.Vector3|null>;
type BoneName='pelvis'|'torso'|'head'|'upperArmL'|'forearmL'|'handL'|'thighL'|'shinL'|'footL'|'upperArmR'|'forearmR'|'handR'|'thighR'|'shinR'|'footR';
type Transform={position:THREE.Vector3;rotation:THREE.Quaternion;valid:boolean};
type Retarget={joints:JointMap;transforms:Record<BoneName,Transform>};
export interface AnatomyRig {
  root:THREE.Group;
  meshes:THREE.SkinnedMesh[];
  jointPositions:JointMap;
  missingParts:number;
  update(pose:Pose3DFrame):void;
  /** Pure sampling; does not advance or mutate the displayed skeleton. */
  sample(pose:Pose3DFrame):JointMap;
  dispose():void;
}

const v=(x:number,y:number,z:number)=>new THREE.Vector3(x,y,z);
// Rest landmarks in the distributed atlas (Y-up, anatomical left +X). The
// pivots were inspected against connected bone components, not guessed from
// muscle bounds. Source dimensions are only a visual body-size reference.
const REST:Record<number,THREE.Vector3>={
  0:v(0,1.61,.035),5:v(.174,1.398,-.025),6:v(-.174,1.398,-.025),
  7:v(.226,1.103,-.033),8:v(-.226,1.103,-.033),
  9:v(.259,.860,.015),10:v(-.259,.860,.015),
  11:v(.075,.865,-.015),12:v(-.075,.865,-.015),
  13:v(.08,.44,-.03),14:v(-.08,.44,-.03),
  15:v(.08,.08,-.035),16:v(-.08,.08,-.035),
  17:v(.085,.028,.10),20:v(-.085,.028,.10),
};
const pelvisRest=v(0,.865,-.015),shoulderRest=v(0,1.398,-.025),headRest=v(0,1.49,-.025);
const REST_TORSO_LENGTH=shoulderRest.distanceTo(pelvisRest);
const SCALE=.56/REST_TORSO_LENGTH;
const names:BoneName[]=['pelvis','torso','head','upperArmL','forearmL','handL','thighL','shinL','footL','upperArmR','forearmR','handR','thighR','shinR','footR'];
const index=new Map(names.map((name,i)=>[name,i]));
const pivot:Record<BoneName,THREE.Vector3>={pelvis:pelvisRest,torso:pelvisRest,head:headRest,
  upperArmL:REST[5],forearmL:REST[7],handL:REST[9],thighL:REST[11],shinL:REST[13],footL:REST[15],
  upperArmR:REST[6],forearmR:REST[8],handR:REST[10],thighR:REST[12],shinR:REST[14],footR:REST[16]};
const sideBones=(left:boolean)=>left?
  {upper:'upperArmL',fore:'forearmL',hand:'handL',thigh:'thighL',shin:'shinL',foot:'footL'} as const:
  {upper:'upperArmR',fore:'forearmR',hand:'handR',thigh:'thighR',shin:'shinR',foot:'footR'} as const;

// All limits below are display constraints in radians. Neither flexion depth
// nor rotation around a limb axis is observable from the image points.
const MAX_ELBOW_FLEXION=2.7, MAX_KNEE_FLEXION=2.6, MAX_FOOT_SWING=.7;

/** Route the rotation through a stable normal to the image instead of taking
 * a new shortest arc directly from rest on every frame. That arc can reverse
 * the entire skin around the bone when an overhead arm passes its antipode. */
function armRotation(rest:THREE.Vector3, direction:THREE.Vector3, parent:THREE.Quaternion, facing:THREE.Vector3) {
  const inherited=rest.clone().normalize().applyQuaternion(parent);
  return new THREE.Quaternion().setFromUnitVectors(facing,direction)
    .multiply(new THREE.Quaternion().setFromUnitVectors(inherited,facing)).multiply(parent);
}

/** Orient the model's transverse axis along the common knee hinge so that
 * the bone surfaces, patella and muscles bend in the same plane as the joints. */
function legOrientation(shaft:THREE.Vector3, hinge:THREE.Vector3) {
  const up=shaft.clone().normalize().negate();
  const side=hinge.clone().projectOnPlane(up).normalize();
  const front=new THREE.Vector3().crossVectors(side,up);
  return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(side,up,front));
}

function retarget(pose:Pose3DFrame):Retarget {
  const joints:JointMap={};for(let i=0;i<65;i++)joints[i]=null;
  // Schematic left/right Z offsets are drawing separation, NOT observations.
  // Feeding them into rotations fabricated body yaw and nose-to-neck pitch.
  const read=(id:number)=>{const p=pose.joints[id];return p?.visible&&p.confidence>=.5&&p.position.every(Number.isFinite)?v(p.position[0],p.position[1],0):null;};
  const transforms={} as Record<BoneName,Transform>;
  for(const name of names)transforms[name]={position:new THREE.Vector3(),rotation:new THREE.Quaternion(),valid:false};
  const l=read(5),r=read(6),lh=read(11),rh=read(12);
  if(!l||!r||!lh||!rh)return{joints,transforms};
  const hip=lh.clone().add(rh).multiplyScalar(.5),observedShoulder=l.clone().add(r).multiplyScalar(.5);
  const y=observedShoulder.clone().sub(hip),torsoLength=y.length();if(torsoLength<.05)return{joints,transforms};y.normalize();
  const shoulderAxis=l.clone().sub(r),hipAxis=lh.clone().sub(rh);
  shoulderAxis.addScaledVector(y,-shoulderAxis.dot(y));hipAxis.addScaledVector(y,-hipAxis.dot(y));
  // A nearly edge-on shoulder pair cannot establish facing. Use the pelvis
  // only if it is resolved; otherwise hide instead of inventing a 90° turn.
  const x=shoulderAxis.length()>=torsoLength*.15?shoulderAxis:hipAxis;
  if(x.length()<torsoLength*.1)return{joints,transforms};x.normalize();
  const z=new THREE.Vector3().crossVectors(x,y).normalize();x.crossVectors(y,z).normalize();
  const base=new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(x,y,z));
  const map=(point:THREE.Vector3)=>point.clone().sub(pelvisRest).multiplyScalar(SCALE).applyQuaternion(base).add(hip);
  transforms.pelvis={position:hip.clone(),rotation:base.clone(),valid:true};
  transforms.torso={position:hip.clone(),rotation:base.clone(),valid:true};
  for(const id of [5,6,11,12])joints[id]=map(REST[id]);
  const headBase=map(headRest),headRotation=base.clone();
  // A nose position says nothing reliable about 3D gaze, especially from the
  // back. Inherit torso facing; only a consistent ear/eye pair can add roll in
  // the image plane. These are conservative animation limits, not measurements.
  const face=[0,1,2,3,4].map(read);
  const plausibleFace=(point:THREE.Vector3|null)=>{
    if(!point)return false;const offset=point.clone().sub(observedShoulder);
    return offset.dot(y)>torsoLength*.08&&offset.dot(y)<torsoLength*.85
      &&Math.abs(offset.dot(x))<torsoLength*.55;
  };
  if(face.some(plausibleFace)){
    for(const [a,b] of [[3,4],[1,2]]){
      const left=face[a],right=face[b];
      if(!plausibleFace(left)||!plausibleFace(right)||pose.joints[a].confidence<.7||pose.joints[b].confidence<.7)continue;
      const axis=left!.clone().sub(right!),span=axis.length();
      if(span<torsoLength*.08||span>torsoLength*.55||axis.dot(x)<=0)continue;
      const roll=THREE.MathUtils.clamp(Math.atan2(axis.dot(y),axis.dot(x)),-.35,.35);
      headRotation.multiply(new THREE.Quaternion().setFromAxisAngle(v(0,0,1),roll));break;
    }
    transforms.head={position:headBase,rotation:headRotation,valid:true};
    joints[0]=REST[0].clone().sub(headRest).multiplyScalar(SCALE).applyQuaternion(headRotation).add(headBase);
  }
  const setSegment=(name:BoneName,a:number,b:number,direction:THREE.Vector3,rotation:THREE.Quaternion)=>{
    const anchor=joints[a]!;
    transforms[name]={position:anchor.clone(),rotation,valid:true};
    joints[b]=anchor.clone().addScaledVector(direction,REST[a].distanceTo(REST[b])*SCALE);
  };
  const arm=(name:BoneName,a:number,b:number,parent:BoneName,proximal?:number)=>{
    const from=read(a),to=read(b);
    if(!from||!to||!joints[a])return;
    const projection=to.sub(from),rest=REST[b].clone().sub(REST[a]);
    const projectedFraction=(projection.length()/torsoLength)/(rest.length()/REST_TORSO_LENGTH);
    // Near end-on, a few pixels of wrist noise can reverse the projected
    // shaft. Taper its influence towards a schematic forward direction.
    const angleFromFacing=Math.PI/2*smooth(0,1,projectedFraction);
    const direction=projection.normalize().multiplyScalar(Math.sin(angleFromFacing)).addScaledVector(z,Math.cos(angleFromFacing));
    if(proximal!==undefined){
      const axis=joints[a]!.clone().sub(joints[proximal]!).normalize(),dot=axis.dot(direction);
      const limit=Math.cos(MAX_ELBOW_FLEXION);
      if(dot<limit){
        // At a closed elbow the two planar clipping directions exchange sides.
        // Bias the perpendicular towards facing as the observed angle closes.
        const tangent=direction.clone().addScaledVector(axis,-dot);
        tangent.addScaledVector(z.clone().projectOnPlane(axis),Math.sin(MAX_ELBOW_FLEXION)-tangent.length()).normalize();
        direction.copy(axis).multiplyScalar(limit).addScaledVector(tangent,Math.sin(MAX_ELBOW_FLEXION));
      }
    }
    setSegment(name,a,b,direction,armRotation(rest,direction,transforms[parent].rotation,z));
  };
  const leg=(bones:ReturnType<typeof sideBones>,h:number,k:number,a:number,t:number)=>{
    const hipPoint=read(h),kneePoint=read(k),anklePoint=read(a);
    if(!hipPoint||!kneePoint)return;
    const thigh=kneePoint.clone().sub(hipPoint);
    if(thigh.length()<torsoLength*.025)return;
    thigh.normalize();
    const restThigh=REST[k].clone().sub(REST[h]),restShin=REST[a].clone().sub(REST[k]);
    const thighBind=legOrientation(restThigh,v(1,0,0)).invert();
    const shin=anklePoint?.clone().sub(kneePoint);
    if(!shin||shin.length()<torsoLength*.025){
      const hinge=z.clone().cross(thigh);
      setSegment(bones.thigh,h,k,thigh,legOrientation(thigh,hinge).multiply(thighBind));
      return;
    }
    shin.normalize();
    const upperLength=restThigh.length()*SCALE,lowerLength=restShin.length()*SCALE;
    const target=thigh.clone().multiplyScalar(upperLength).addScaledVector(shin,lowerLength);
    const axis=target.clone().normalize();
    const minReach=Math.sqrt(upperLength**2+lowerLength**2+2*upperLength*lowerLength*Math.cos(MAX_KNEE_FLEXION));
    const reach=THREE.MathUtils.clamp(target.length(),minReach,upperLength+lowerLength);
    const along=(upperLength**2-lowerLength**2+reach**2)/(2*reach);
    const height=Math.sqrt(Math.max(0,upperLength**2-along**2));
    // Keep the original reachable ankle target; solve the fixed-length knee
    // forward of the hip–ankle line. A bounded turn towards the observed
    // knee permits leg turnout without allowing a sideways/reversed hinge.
    const side=z.clone().cross(axis);
    const turnout=THREE.MathUtils.clamp(thigh.dot(side),-.55,.55);
    const pole=z.clone().multiplyScalar(Math.sqrt(1-turnout**2)).addScaledVector(side,turnout);
    const hinge=pole.clone().cross(axis);
    const kneeOffset=axis.clone().multiplyScalar(along).addScaledVector(pole,height);
    const upper=kneeOffset.clone().normalize();
    const lower=axis.multiplyScalar(reach).sub(kneeOffset).normalize();
    setSegment(bones.thigh,h,k,upper,legOrientation(upper,hinge).multiply(thighBind));
    setSegment(bones.shin,k,a,lower,legOrientation(lower,hinge).multiply(legOrientation(restShin,v(1,0,0)).invert()));

    const toe=read(t);
    if(!toe)return;
    const observed=toe.sub(anklePoint!);
    if(observed.length()<torsoLength*.025)return;
    const parent=transforms[bones.shin].rotation;
    const rest=REST[t].clone().sub(REST[a]).normalize().applyQuaternion(parent);
    const swing=new THREE.Quaternion().setFromUnitVectors(rest,observed.normalize());
    const rotation=new THREE.Quaternion().rotateTowards(swing,MAX_FOOT_SWING);
    setSegment(bones.foot,a,t,rest.applyQuaternion(rotation),rotation.multiply(parent));
  };
  for(const left of [true,false]){
    const b=sideBones(left),s=left?5:6,e=left?7:8,w=left?9:10,h=left?11:12,k=left?13:14,a=left?15:16,t=left?17:20;
    arm(b.upper,s,e,'torso');arm(b.fore,e,w,b.upper,s);
    leg(b,h,k,a,t);
    if(joints[w])transforms[b.hand]={position:joints[w]!.clone(),rotation:transforms[b.fore].rotation.clone(),valid:true};
  }
  return{joints,transforms};
}

/** Classify rigid bone components, never split an individual bone at a joint. */
function regionBone(point:THREE.Vector3):BoneName {
  const b=sideBones(point.x>=0),x=Math.abs(point.x),y=point.y;
  if(y>1.50)return'head';
  if(y>1.04&&x>.16)return b.upper;
  if(y>.84&&y<=1.13&&x>.195)return b.fore;
  if(y>.63&&y<=.88&&x>.19)return b.hand;
  if(y<.105)return b.foot;
  if(y<.45)return b.shin;
  if(y<.82)return b.thigh;
  if(y<1.01)return'pelvis';
  return'torso';
}
const smooth=(a:number,b:number,x:number)=>{const t=THREE.MathUtils.clamp((x-a)/(b-a),0,1);return t*t*(3-2*t);};
type Weight=[BoneName,number][];
function weights(point:THREE.Vector3,data:Record<string,unknown>):Weight {
  const b=sideBones(data.side?data.side==='left':point.x>=0),y=point.y;
  const blend=(a:BoneName,c:BoneName,t:number):Weight=>[[a,1-t],[c,t]];
  switch(data.region){
    case 'shoulder':case 'arms': {
      // Broad shoulder attachments remain on the torso while the belly follows
      // the humerus. Skinning is geometric deformation, not muscle activation.
      if(y>1.32)return blend(b.upper,'torso',smooth(1.32,1.405,y)*.72);
      return blend(b.upper,b.fore,1-smooth(1.075,1.15,y));
    }
    case 'forearms':
      if(y>1.07)return blend(b.fore,b.upper,smooth(1.07,1.17,y));
      return blend(b.fore,b.hand,1-smooth(.82,.89,y));
    case 'hips':return blend(b.thigh,'pelvis',smooth(.79,.94,y));
    case 'thigh':
      if(y>.80)return blend(b.thigh,'pelvis',smooth(.80,.94,y));
      return blend(b.thigh,b.shin,1-smooth(.40,.49,y));
    case 'calves':
      if(y>.40)return blend(b.shin,b.thigh,smooth(.40,.49,y));
      return blend(b.shin,b.foot,1-smooth(.055,.12,y));
    case 'neck':return blend('torso','head',smooth(1.44,1.54,y));
    case 'back':
      // The atlas labels the trapezius as back, including its skull attachment.
      // Let that attachment follow head roll instead of stretching the neck.
      if(y>1.44)return blend('torso','head',smooth(1.44,1.54,y));
      return blend('pelvis','torso',smooth(.88,1.06,y));
    case 'chest':case 'core':return blend('pelvis','torso',smooth(.88,1.06,y));
    default:return[[regionBone(point),1]];
  }
}

function componentBones(geometry:THREE.BufferGeometry):BoneName[] {
  const position=geometry.getAttribute('position'),n=position.count,parent=Int32Array.from({length:n},(_,i)=>i);
  const find=(a:number):number=>{while(parent[a]!==a){parent[a]=parent[parent[a]];a=parent[a];}return a;};
  const union=(a:number,b:number)=>{const ra=find(a),rb=find(b);if(ra!==rb)parent[rb]=ra;};
  const count=geometry.index?.count??n,at=(i:number)=>geometry.index?geometry.index.getX(i):i;
  for(let i=0;i<count;i+=3){union(at(i),at(i+1));union(at(i),at(i+2));}
  const sums=new Map<number,{point:THREE.Vector3;count:number}>();
  for(let i=0;i<n;i++){const root=find(i),sum=sums.get(root)??{point:new THREE.Vector3(),count:0};sum.point.add(v(position.getX(i),position.getY(i),position.getZ(i)));sum.count++;sums.set(root,sum);}
  const boneByRoot=new Map<number,BoneName>();for(const [id,sum] of sums)boneByRoot.set(id,regionBone(sum.point.divideScalar(sum.count)));
  return Array.from({length:n},(_,i)=>boneByRoot.get(find(i))!);
}

/** Compact a material's triangles into a separately hideable anatomical part. */
function subset(source:THREE.BufferGeometry,triangles:number[]):THREE.BufferGeometry {
  const remap=new Map<number,number>(),vertices:number[]=[],indices:number[]=[];
  for(const old of triangles){let next=remap.get(old);if(next===undefined){next=vertices.length;vertices.push(old);remap.set(old,next);}indices.push(next);}
  const geometry=new THREE.BufferGeometry();
  for(const [name,attribute] of Object.entries(source.attributes)){
    const array=new Float32Array(vertices.length*attribute.itemSize);
    for(let i=0;i<vertices.length;i++)for(let c=0;c<attribute.itemSize;c++)array[i*attribute.itemSize+c]=attribute.getComponent(vertices[i],c);
    geometry.setAttribute(name,new THREE.BufferAttribute(array,attribute.itemSize,attribute.normalized));
  }
  geometry.setIndex(indices);return geometry;
}

/**
 * Bind the atlas to a small shared GPU skeleton. Bone surfaces use one rigid
 * influence per connected component; muscles have smooth local joint weights.
 * The fixed model proportions follow observed 2D segment directions. This is
 * anatomical animation retargeting, not measured 3D reconstruction or force.
 */
export function createAnatomyRig(source:THREE.Group):AnatomyRig {
  const root=new THREE.Group();root.name='anatomical-motion-rig';
  const bones=names.map(name=>{const bone=new THREE.Bone();bone.name=name;bone.position.copy(pivot[name]);root.add(bone);return bone;});
  root.updateMatrixWorld(true);const skeleton=new THREE.Skeleton(bones);skeleton.calculateInverses();
  const meshes:THREE.SkinnedMesh[]=[],required=new Map<THREE.SkinnedMesh,BoneName[]>(),geometries=new Set<THREE.BufferGeometry>();
  const add=(geometry:THREE.BufferGeometry,original:THREE.Mesh,rigid?:BoneName)=>{
    const position=geometry.getAttribute('position'),skinIndex=new Uint16Array(position.count*4),skinWeight=new Float32Array(position.count*4),used=new Set<BoneName>();
    // Long fascia/tendons span joints. Keep bones rigid but let connective
    // surfaces share the adjacent muscle weights so both attachments follow.
    const connective=original.userData.contextType==='connective';
    const softRegion=rigid?.startsWith('thigh')?'thigh':rigid?.startsWith('shin')?'calves':rigid?.startsWith('upperArm')?'arms':rigid?.startsWith('forearm')?'forearms':rigid==='pelvis'||rigid==='torso'?'core':undefined;
    const weightData=connective&&softRegion?{...original.userData,region:softRegion}:original.userData;
    for(let i=0;i<position.count;i++){
      const influences:Weight=rigid&&!(connective&&softRegion)?[[rigid,1]]:weights(v(position.getX(i),position.getY(i),position.getZ(i)),weightData);
      influences.filter(([,weight])=>weight>1e-5).forEach(([name,weight],slot)=>{skinIndex[i*4+slot]=index.get(name)!;skinWeight[i*4+slot]=weight;used.add(name);});
    }
    geometry.setAttribute('skinIndex',new THREE.Uint16BufferAttribute(skinIndex,4));geometry.setAttribute('skinWeight',new THREE.Float32BufferAttribute(skinWeight,4));
    geometries.add(geometry);
    const mesh=new THREE.SkinnedMesh(geometry,original.material);mesh.name=original.name+(rigid?'__'+rigid:'');
    mesh.userData={...original.userData,poseVisible:true,rigBone:rigid,rigRegion:rigid?.includes('Arm')?'upperArm':rigid?.includes('forearm')||rigid?.includes('hand')?'forearm':rigid?.includes('thigh')||rigid?.includes('shin')||rigid?.includes('foot')?'legs':'trunk'};
    mesh.frustumCulled=false;root.add(mesh);mesh.bind(skeleton,new THREE.Matrix4());meshes.push(mesh);required.set(mesh,[...used]);
  };
  source.updateMatrixWorld(true);
  source.traverse(object=>{
    if(!(object instanceof THREE.Mesh))return;
    const geometry=object.geometry.clone();geometry.applyMatrix4(object.matrixWorld);
    if(object.userData.contextType==='bone'||object.userData.supportId){
      const position=geometry.getAttribute('position'),component=componentBones(geometry),buckets=new Map<BoneName,number[]>();
      const count=geometry.index?.count??position.count,at=(i:number)=>geometry.index?geometry.index.getX(i):i;
      for(let i=0;i<count;i+=3){const a=at(i),b=at(i+1),c=at(i+2),bone=component[a];const triangles=buckets.get(bone)??[];triangles.push(a,b,c);buckets.set(bone,triangles);}
      for(const [bone,triangles] of buckets)add(subset(geometry,triangles),object,bone);
      geometry.dispose();
    }else add(geometry,object);
  });
  const rig:AnatomyRig={root,meshes,jointPositions:{},missingParts:0,
    sample:pose=>retarget(pose).joints,
    update(pose){
      const state=retarget(pose);Object.assign(rig.jointPositions,state.joints);rig.missingParts=0;
      for(let i=0;i<names.length;i++){
        const transform=state.transforms[names[i]],bone=bones[i];bone.position.copy(transform.position);bone.quaternion.copy(transform.rotation);bone.scale.setScalar(SCALE);
        if(!transform.valid)rig.missingParts++;
      }
      for(const mesh of meshes){mesh.userData.poseVisible=required.get(mesh)!.every(name=>state.transforms[name].valid);mesh.visible=mesh.userData.poseVisible;}
      root.updateMatrixWorld(true);skeleton.update();
    },
    dispose(){geometries.forEach(g=>g.dispose());skeleton.dispose();root.clear();},
  };
  return rig;
}
