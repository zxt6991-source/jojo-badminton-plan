import {useEffect,useLayoutEffect,useMemo,useRef,useState,type CSSProperties} from 'react';
import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import type {Analysis,SwingEvent,VideoAsset} from '@badminton/schema';
import {REGION_COLORS,type BodyRegion} from '@badminton/metrics';
import {createPose3DSequence} from './pose3d';
import {createAnatomyRig,type AnatomyRig} from './anatomyRig';
import {describeAnatomyPart,type AnatomyPart} from './anatomyCatalog';
import Icon from './Icon';
import './body3d.css';

type Focus=BodyRegion|'all';
type CameraView='camera'|'oblique'|'back';
type Layer='reference'|'muscles'|'skeleton';
interface Props {
  video:VideoAsset;analysis:Analysis;event:SwingEvent;points:Float32Array;timestamps:Float32Array;
  frame:number;onSeek:(frame:number)=>void;onFrame:(frame:number)=>void;seekToken:number;
  focus:Focus;onFocus:(region:Focus)=>void;demo:boolean;
}
interface SceneApi {update:()=>void;preset:(view:CameraView)=>void;zoom:(factor:number)=>void}
const regionNames:Record<BodyRegion,string>={legs:'下肢',trunk:'躯干',upperArm:'上臂',forearm:'前臂'};
const up=new THREE.Vector3(0,1,0);
function closestFrame(times:Float32Array,time:number){let lo=0,hi=times.length-1;while(lo<hi){const mid=(lo+hi)>>>1;if(times[mid]<time)lo=mid+1;else hi=mid;}return lo>0&&Math.abs(times[lo-1]-time)<Math.abs(times[lo]-time)?lo-1:lo;}

export default function Body3DViewer(props:Props){
  const {video,analysis,event,points,timestamps,frame,onSeek,onFrame,seekToken,focus,onFocus,demo}=props;
  const host=useRef<HTMLDivElement>(null),runtime=useRef<SceneApi|null>(null),latest=useRef(props);latest.current=props;
  const [playing,setPlaying]=useState(false),[speed,setSpeed]=useState(.5),[layer,setLayer]=useState<Layer>('reference');
  const [trail,setTrail]=useState(true),[racket,setRacket]=useState(true),[opacity,setOpacity]=useState(1),[loop,setLoop]=useState(true);
  const [cameraView,setCameraView]=useState<CameraView>('oblique'),[error,setError]=useState(''),[ready,setReady]=useState(false),[attempt,setAttempt]=useState(0);
  const [selected,setSelected]=useState<AnatomyPart|null>(null);
  const visual=useRef({layer,trail,racket,opacity,selected});visual.current={layer,trail,racket,opacity,selected};
  const sequence=useMemo(()=>createPose3DSequence(video,points),[video,points]);
  const time=timestamps[frame]??0,duration=timestamps[timestamps.length-1]??video.durationSec;
  const currentPose=useMemo(()=>sequence.frame(frame),[sequence,frame]);
  const start=event.phases.backswingStartFrame??Math.max(0,event.impact.frame-Math.round(video.captureFps*.5));
  const end=event.phases.followThroughEndFrame??Math.min(timestamps.length-1,event.impact.frame+Math.round(video.captureFps*.5));

  useLayoutEffect(()=>{setPlaying(false);},[seekToken,analysis.id,event.id]);
  useEffect(()=>{setSelected(p=>p&&p.region!==focus?null:p);},[focus]);
  useEffect(()=>{setSelected(null);},[event.id]);
  useEffect(()=>{
    if(!playing||error||!ready)return;
    const generation={seekToken:latest.current.seekToken,analysisId:latest.current.analysis.id,eventId:latest.current.event.id};
    let active=true,handle=0,previous=performance.now(),clock=latest.current.timestamps[latest.current.frame]??0;
    const tick=(now:number)=>{
      const current=latest.current;if(!active)return;
      if(current.seekToken!==generation.seekToken||current.analysis.id!==generation.analysisId||current.event.id!==generation.eventId){setPlaying(false);return;}
      clock+=Math.min(.15,(now-previous)/1000)*speed;previous=now;
      const t=current.timestamps,limit=loop?t[end]:t[t.length-1];
      if(clock>=limit){if(loop)clock=t[start]+((clock-t[start])%Math.max(.001,t[end]-t[start]));else{current.onFrame(t.length-1);setPlaying(false);return;}}
      current.onFrame(closestFrame(t,clock));handle=requestAnimationFrame(tick);
    };handle=requestAnimationFrame(tick);return()=>{active=false;cancelAnimationFrame(handle);};
  },[playing,speed,error,ready,loop,start,end]);
  useEffect(()=>{runtime.current?.update();},[frame,focus,layer,trail,racket,opacity,selected,event]);

  useEffect(()=>{
    const container=host.current;if(!container)return;
    setReady(false);setError('');setPlaying(false);setSelected(null);
    const abort=new AbortController();let disposed=false,lost=false,raf=0,rig:AnatomyRig|null=null;
    let renderer:THREE.WebGLRenderer;
    try{renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:'high-performance'});}
    catch{setError('当前设备无法启动 3D。可以切换到视频 / 2D 对照继续查看动作。');return;}
    const canvas=renderer.domElement;canvas.dataset.viewer='body-3d';canvas.dataset.model='anatomy';canvas.tabIndex=0;canvas.setAttribute('role','img');container.appendChild(canvas);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,1.75));renderer.outputColorSpace=THREE.SRGBColorSpace;
    renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.1;
    const scene=new THREE.Scene();scene.background=new THREE.Color('#fafaf8');
    const height=Math.max(1.8,sequence.height),center=new THREE.Vector3(...sequence.center);
    const camera=new THREE.PerspectiveCamera(34,1,.02,40);
    const controls=new OrbitControls(camera,canvas);controls.target.copy(center);controls.enableDamping=true;controls.dampingFactor=.09;
    controls.enablePan=true;controls.screenSpacePanning=true;controls.minDistance=.5;controls.maxDistance=height*4.5;controls.minPolarAngle=.1;controls.maxPolarAngle=Math.PI-.1;controls.rotateSpeed=.65;
    scene.add(new THREE.HemisphereLight('#ffffff','#b0a69b',1.15));
    const key=new THREE.DirectionalLight('#fffaf1',2.6);key.position.set(-3,5,5);scene.add(key);
    const fill=new THREE.DirectionalLight('#ffffff',.7);fill.position.set(3,2,1);scene.add(fill);
    const rim=new THREE.DirectionalLight('#ffffff',1.3);rim.position.set(0,3,-4);scene.add(rim);
    const geometries=new Set<THREE.BufferGeometry>(),materials=new Set<THREE.Material>();
    const geo=<T extends THREE.BufferGeometry>(g:T)=>{geometries.add(g);return g;};
    const mat=<T extends THREE.Material>(m:T)=>{materials.add(m);return m;};
    const parts:{mesh:THREE.SkinnedMesh;part:AnatomyPart;material:THREE.MeshStandardMaterial}[]=[];
    const halo=new THREE.Mesh(geo(new THREE.TorusGeometry(.065,.004,8,40)),mat(new THREE.MeshBasicMaterial({color:'#c49236'})));scene.add(halo);
    const trailPositions=new Float32Array(256*6),trailGeometry=geo(new THREE.BufferGeometry());trailGeometry.setAttribute('position',new THREE.BufferAttribute(trailPositions,3));trailGeometry.setDrawRange(0,0);
    const trailLines=new THREE.LineSegments(trailGeometry,mat(new THREE.LineBasicMaterial({color:'#aa7259',transparent:true,opacity:.65})));trailLines.frustumCulled=false;scene.add(trailLines);
    const racketGroup=new THREE.Group();scene.add(racketGroup);
    const handle=new THREE.Mesh(geo(new THREE.CylinderGeometry(.012,.014,.12,10)),mat(new THREE.MeshStandardMaterial({color:'#656e63',roughness:.8})));handle.position.y=.06;racketGroup.add(handle);
    const shaft=new THREE.Mesh(geo(new THREE.CylinderGeometry(.004,.004,.18,8)),mat(new THREE.MeshStandardMaterial({color:'#94a08b',roughness:.6})));shaft.position.y=.21;racketGroup.add(shaft);
    const rimMesh=new THREE.Mesh(geo(new THREE.TorusGeometry(.1,.006,8,40)),mat(new THREE.MeshStandardMaterial({color:'#6b7c58',roughness:.6})));rimMesh.scale.y=1.3;rimMesh.position.y=.42;racketGroup.add(rimMesh);
    const strings:number[]=[];for(let i=-4;i<=4;i++){const a=i*.019,b=Math.sqrt(.01-a*a);strings.push(a,.42-b*1.3,0,a,.42+b*1.3,0,-b,.42+a*1.3,0,b,.42+a*1.3,0);}
    const stringGeo=geo(new THREE.BufferGeometry());stringGeo.setAttribute('position',new THREE.Float32BufferAttribute(strings,3));racketGroup.add(new THREE.LineSegments(stringGeo,mat(new THREE.LineBasicMaterial({color:'#a6b09e',transparent:true,opacity:.6}))));
    const wristCache=new Map<number,{left:THREE.Vector3|null;right:THREE.Vector3|null}>();
    function collectWrist(f:number){if(!rig)return;const joints=rig.sample(sequence.frame(f));wristCache.set(f,{left:joints[9]?.clone()??null,right:joints[10]?.clone()??null});}
    const update=()=>{
      if(!rig||disposed)return;
      const p=latest.current,v=visual.current,pose=sequence.frame(p.frame);
      const wristId=p.event.handedness==='left'?9:p.event.handedness==='right'?10:-1;
      let first=p.frame;while(first>0&&p.timestamps[p.frame]-p.timestamps[first-1]<=.2)first--;
      if(v.trail&&wristId>=0)for(let f=first;f<=p.frame;f++)if(!wristCache.has(f))collectWrist(f);
      rig.update(pose);
      let visibleMuscles=0;
      for(const {mesh,part,material} of parts){
        const regionMatch=p.focus==='all'||p.focus===part.region;
        const layerVisible=part.kind==='bone'||v.layer!=='skeleton'&&(p.focus!=='all'?regionMatch:v.layer==='muscles'||part.referenceVisible);
        mesh.visible=mesh.userData.poseVisible!==false&&layerVisible;
        if(part.kind==='muscle'&&mesh.visible)visibleMuscles++;
        const chosen=v.selected?.id===part.id;
        material.color.set(part.kind==='bone'?'#d4cba0':part.kind==='connective'?'#ded5be':chosen?'#bb574c':'#cf7c74');
        material.emissive.set(chosen?'#70251d':0);material.emissiveIntensity=chosen?.1:0;
        material.opacity=part.kind==='bone'?1:v.opacity;material.transparent=material.opacity<1;material.depthWrite=material.opacity>=.98;
      }
      const wrist=wristId<0?null:rig.jointPositions[wristId],elbow=wristId<0?null:rig.jointPositions[wristId-2];
      halo.visible=!!wrist&&Math.abs(p.frame-p.event.impact.frame)<=1;if(wrist){halo.position.copy(wrist);halo.quaternion.copy(camera.quaternion);}
      racketGroup.visible=v.racket&&!!wrist&&!!elbow;
      if(wrist&&elbow){racketGroup.position.copy(wrist);const axis=wrist.clone().sub(elbow).normalize();racketGroup.quaternion.setFromUnitVectors(up,axis);}
      let count=0;trailLines.visible=v.trail&&wristId>=0&&(p.focus==='all'||p.focus==='forearm');
      if(trailLines.visible){const side=wristId===9?'left':'right';for(let f=first+1;f<=p.frame&&count<256;f++){
        if(p.timestamps[f]-p.timestamps[f-1]>.05)continue;const a=wristCache.get(f-1)?.[side],b=wristCache.get(f)?.[side];if(!a||!b)continue;
        a.toArray(trailPositions,count*6);b.toArray(trailPositions,count*6+3);count++;
      }}trailGeometry.setDrawRange(0,count*2);trailGeometry.attributes.position.needsUpdate=true;
      canvas.dataset.frame=String(p.frame);canvas.dataset.focus=p.focus;canvas.dataset.visibleJoints=String(pose.visibleCount);canvas.dataset.style=v.layer==='skeleton'?'skeleton':'solid';canvas.dataset.layer=v.layer;canvas.dataset.visibleMuscles=String(visibleMuscles);
      canvas.dataset.wrist=wrist?`${wrist.x.toFixed(4)},${wrist.y.toFixed(4)},${wrist.z.toFixed(4)}`:'missing';
      canvas.setAttribute('aria-label',`骨骼与肌肉 3D 动作回放，第 ${p.frame} 帧，${p.focus==='all'?'全身':regionNames[p.focus]}`);
    };
    const preset=(view:CameraView)=>{const d=height*1.85*Math.max(1,.8/camera.aspect);camera.position.copy(center).add(view==='camera'?new THREE.Vector3(0,height*.03,d):view==='back'?new THREE.Vector3(-d*.35,height*.12,-d*.94):new THREE.Vector3(d*.35,height*.12,d*.94));controls.target.copy(center);controls.update();};
    const zoom=(factor:number)=>{const offset=camera.position.clone().sub(controls.target);offset.setLength(THREE.MathUtils.clamp(offset.length()*factor,controls.minDistance,controls.maxDistance));camera.position.copy(controls.target).add(offset);controls.update();};
    runtime.current={update,preset,zoom};
    const resize=()=>{const rect=container.getBoundingClientRect();if(!rect.width||!rect.height)return;const old=camera.aspect;renderer.setSize(rect.width,rect.height,false);camera.aspect=rect.width/rect.height;camera.updateProjectionMatrix();if(Math.abs(old-camera.aspect)>.15){preset('oblique');setCameraView('oblique');}};
    const observer=new ResizeObserver(resize);observer.observe(container);resize();preset('oblique');setCameraView('oblique');
    const animate=()=>{if(disposed||lost)return;controls.update();if(halo.visible)halo.quaternion.copy(camera.quaternion);renderer.render(scene,camera);raf=requestAnimationFrame(animate);};animate();
    const onLost=(e:Event)=>{e.preventDefault();lost=true;cancelAnimationFrame(raf);setReady(false);setPlaying(false);setError('3D 图形连接中断，请重新加载，或切换到视频 / 2D 对照。');};canvas.addEventListener('webglcontextlost',onLost);
    const raycaster=new THREE.Raycaster(),pointer=new THREE.Vector2();let down:{x:number;y:number}|null=null;
    const pointerDown=(e:PointerEvent)=>{down={x:e.clientX,y:e.clientY};};
    const pointerUp=(e:PointerEvent)=>{if(!down||Math.hypot(e.clientX-down.x,e.clientY-down.y)>5){down=null;return;}down=null;
      const rect=canvas.getBoundingClientRect();pointer.set((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1);raycaster.setFromCamera(pointer,camera);
      const pickable=parts.filter(o=>o.mesh.visible&&o.part.kind==='muscle').map(o=>o.mesh);
      // SkinnedMesh caches its bounds; refresh on picking after the pose moves.
      for(const mesh of pickable)mesh.computeBoundingSphere();
      const hit=raycaster.intersectObjects(pickable,false)[0];if(!hit)return;
      const item=parts.find(o=>o.mesh===hit.object);if(!item)return;setPlaying(false);setSelected(item.part); 
      if(item.part.region)latest.current.onFocus(item.part.region);
    };canvas.addEventListener('pointerdown',pointerDown);canvas.addEventListener('pointerup',pointerUp);
    const keydown=(e:KeyboardEvent)=>{if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','-','=','0'].includes(e.key))return;e.preventDefault();
      if(e.key==='0'){preset('oblique');setCameraView('oblique');return;}if(e.key==='+'||e.key==='='){zoom(.85);return;}if(e.key==='-'){zoom(1.15);return;}
      const spherical=new THREE.Spherical().setFromVector3(camera.position.clone().sub(controls.target));spherical.theta+=e.key==='ArrowLeft'?-.12:e.key==='ArrowRight'?.12:0;spherical.phi=THREE.MathUtils.clamp(spherical.phi+(e.key==='ArrowUp'?-.1:e.key==='ArrowDown'?.1:0),.1,Math.PI-.1);camera.position.copy(controls.target).add(new THREE.Vector3().setFromSpherical(spherical));controls.update();};canvas.addEventListener('keydown',keydown);
    const disposeSource=(source:THREE.Group)=>{source.traverse(o=>{if(o instanceof THREE.Mesh){o.geometry.dispose();(Array.isArray(o.material)?o.material:[o.material]).forEach(m=>m.dispose());}});};
    fetch('/models/anatomy/body.glb',{signal:abort.signal}).then(r=>{if(!r.ok)throw new Error('model unavailable');return r.arrayBuffer();})
      .then(b=>disposed?null:new GLTFLoader().parseAsync(b,''))
      .then(gltf=>{
        if(!gltf)return;if(disposed){disposeSource(gltf.scene);return;}
        rig=createAnatomyRig(gltf.scene);
        for(const mesh of rig.meshes){(Array.isArray(mesh.material)?mesh.material:[mesh.material]).forEach(m=>m.dispose());const material=mat(new THREE.MeshStandardMaterial({roughness:.78,metalness:0,side:THREE.DoubleSide}));mesh.material=material;parts.push({mesh,part:describeAnatomyPart(mesh.userData,mesh.name),material});}
        disposeSource(gltf.scene);scene.add(rig.root);update();canvas.dataset.skinnedMeshes=String(rig.meshes.length);if(!lost)setReady(true);
      }).catch(cause=>{if(disposed||cause?.name==='AbortError')return;setPlaying(false);setError('人体动作模型加载失败，请重新加载。也可切换到视频 / 2D 对照。');});
    return()=>{disposed=true;abort.abort();cancelAnimationFrame(raf);observer.disconnect();controls.dispose();runtime.current=null;
      canvas.removeEventListener('webglcontextlost',onLost);canvas.removeEventListener('pointerdown',pointerDown);canvas.removeEventListener('pointerup',pointerUp);canvas.removeEventListener('keydown',keydown);
      rig?.dispose();geometries.forEach(g=>g.dispose());materials.forEach(m=>m.dispose());renderer.dispose();renderer.forceContextLoss();canvas.remove();};
  },[sequence,attempt]);

  function seek(next:number){setPlaying(false);onSeek(Math.max(0,Math.min(timestamps.length-1,next)));}
  function toggle(){if(loop&&(frame<start||frame>=end))onFrame(start);else if(frame>=timestamps.length-1)onFrame(0);setPlaying(v=>!v);}
  function chooseView(view:CameraView){setCameraView(view);runtime.current?.preset(view);}
  return <section className="body3d-card" aria-label="3D 身体运动视图">
    <div className="body3d-heading"><span><Icon name="bone" size={18}/>骨骼与肌肉 · 动作回放</span><span className="body3d-flag">{demo?'示意 3D（合成数据）':'示意 3D · 二维姿态驱动'}</span></div>
    <div className="body3d-layers" aria-label="动作模型显示层">{(['reference','muscles','skeleton'] as Layer[]).map(value=><button key={value} aria-pressed={layer===value} onClick={()=>setLayer(value)}>{{reference:'骨骼 + 主要肌群',muscles:'全部肌肉',skeleton:'仅骨骼'}[value]}</button>)}</div>
    <div className="body3d-stage"><div className="body3d-canvas" ref={host}/>
      {!ready&&!error&&<div className="body3d-loading" role="status"><span className="spinner"/>正在绑定骨骼与肌肉动作…</div>}
      {error&&<div className="body3d-error" role="alert"><Icon name="info" size={26}/><strong>3D 视图暂不可用</strong><p>{error}</p><button onClick={()=>setAttempt(a=>a+1)}>重新加载</button></div>}
      <div className="body3d-stage-top"><span className="body3d-live"><i/>{demo?'合成挥拍演示':'跟随视频姿态'}</span><span className="body3d-frame">{String(frame).padStart(4,'0')} FRAME</span></div>
      <div className="body3d-camera-presets" aria-label="3D 观察方向">{(['camera','oblique','back'] as CameraView[]).map(view=><button key={view} disabled={!ready} aria-pressed={cameraView===view} className={cameraView===view?'selected':''} onClick={()=>chooseView(view)}>{{camera:'摄像机方向',oblique:'斜侧',back:'背侧'}[view]}</button>)}</div>
      <div className="body3d-view-tools"><button aria-label="放大 3D 人体" disabled={!ready} onClick={()=>runtime.current?.zoom(.85)}>+</button><button aria-label="缩小 3D 人体" disabled={!ready} onClick={()=>runtime.current?.zoom(1.15)}>−</button><button aria-label="重置 3D 视角" disabled={!ready} onClick={()=>chooseView('oblique')}><Icon name="refresh" size={16}/></button></div>
      {selected&&<div className="body3d-selected">{selected.label}</div>}
      <div className="body3d-stage-bottom"><span>拖动旋转 · 滚轮 / 双指缩放</span><div><button aria-pressed={racket} onClick={()=>setRacket(v=>!v)}>球拍示意</button><button aria-pressed={trail} onClick={()=>setTrail(v=>!v)}>手腕轨迹</button></div></div>
    </div>
    <div className="body3d-transport"><div><button className="play-button" aria-label={playing?'暂停 3D 动作':'播放 3D 动作'} onClick={toggle} disabled={!ready}><Icon name={playing?'pause':'play'} size={17}/></button><button className="icon-button" aria-label="3D 上一帧" onClick={()=>seek(frame-1)}><Icon name="prev" size={16}/></button><button className="icon-button" aria-label="3D 下一帧" onClick={()=>seek(frame+1)}><Icon name="next" size={16}/></button><span className="body3d-time">{time.toFixed(2)}<small> / {duration.toFixed(2)} s</small></span></div><select aria-label="3D 播放速度" value={speed} onChange={e=>setSpeed(Number(e.target.value))}><option value={.25}>0.25× 慢放</option><option value={.5}>0.5× 慢放</option><option value={1}>1× 原速</option></select></div>
    <div className="body3d-scrubber"><input type="range" min={0} max={timestamps.length-1} value={frame} aria-label="3D 动作时间轴" onChange={e=>seek(Number(e.target.value))}/></div>
    <div className="body3d-adjust"><button aria-pressed={loop} onClick={()=>setLoop(v=>!v)}><Icon name="refresh" size={13}/>循环这一拍</button><label>肌肉透明度<input aria-label="肌肉透明度" type="range" min={15} max={100} value={opacity*100} onChange={e=>setOpacity(Number(e.target.value)/100)} disabled={layer==='skeleton'}/></label></div>
    <div className="body3d-regions" aria-label="3D 身体区域"><button aria-pressed={focus==='all'} className={focus==='all'?'active':''} onClick={()=>onFocus('all')}>全身</button>{(Object.keys(regionNames) as BodyRegion[]).map(id=><button key={id} style={{'--part-color':REGION_COLORS[id]} as CSSProperties} className={focus===id?'active':''} aria-pressed={focus===id} onClick={()=>onFocus(focus===id?'all':id)}><i/>{regionNames[id]}</button>)}</div>
    <p className="body3d-note"><Icon name="info" size={14}/><span>按解剖分组着色，不表示激活强度。头部转向、前后深度及球拍位置为示意。{[5,6,7,8,9,10,11,12,13,14,15,16].some(id=>!currentPose.joints[id].visible)?' 遮挡部位已隐藏。':''}</span></p>
    <details className="body3d-credits"><summary>模型来源</summary><p>模型源自 <a href="https://github.com/slfresh/fitmitwith-anatomy-atlas" target="_blank" rel="noreferrer">Fit Mit With / Z-Anatomy / BodyParts3D</a>（CC BY-SA 4.0），加入关节绑定、蒙皮权重及动作映射。<a href="/models/anatomy/ATTRIBUTION.txt" target="_blank" rel="noreferrer">原始署名</a> · <a href="/models/anatomy/MOTION-NOTICE.txt" target="_blank" rel="noreferrer">动作模型说明</a></p></details>
  </section>;
}
