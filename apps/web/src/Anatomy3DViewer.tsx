import {useEffect,useRef,useState,type CSSProperties} from 'react';
import * as THREE from 'three';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {REGION_COLORS,type BodyRegion} from '@badminton/metrics';
import {describeAnatomyPart,REGION_ANATOMY_COPY,type AnatomyPart} from './anatomyCatalog';
import Icon from './Icon';
import './anatomy3d.css';

type Focus=BodyRegion|'all';
type Layer='reference'|'muscles'|'skeleton';
type View='front'|'side'|'back';
interface Props {focus:Focus;onFocus:(focus:Focus)=>void}
interface Runtime {update:()=>void;view:(view:View)=>void;zoom:(factor:number)=>void;focus:()=>void}
interface PartMesh {mesh:THREE.Mesh;part:AnatomyPart;material:THREE.MeshStandardMaterial}
const regionNames:Record<BodyRegion,string>={legs:'下肢',trunk:'躯干',upperArm:'肩与上臂',forearm:'前臂'};
const MODEL_URL='/models/anatomy/body.glb';

export default function Anatomy3DViewer({focus,onFocus}:Props){
  const host=useRef<HTMLDivElement>(null),runtime=useRef<Runtime|null>(null);
  const [layer,setLayer]=useState<Layer>('reference'),[view,setView]=useState<View>('front');
  const [opacity,setOpacity]=useState(1),[selected,setSelected]=useState<AnatomyPart|null>(null);
  const [parts,setParts]=useState<AnatomyPart[]>([]),[status,setStatus]=useState<'loading'|'ready'|'error'>('loading');
  const [error,setError]=useState(''),[attempt,setAttempt]=useState(0);
  const latest=useRef({focus,onFocus,layer,opacity,selected});latest.current={focus,onFocus,layer,opacity,selected};

  useEffect(()=>{runtime.current?.update();},[focus,layer,opacity,selected]);
  // A region chosen elsewhere replaces a previous individual-muscle selection.
  useEffect(()=>{setSelected(p=>p&&p.region!==focus?null:p);},[focus]);

  useEffect(()=>{
    const container=host.current;if(!container)return;
    let disposed=false,lost=false,raf=0;
    const abort=new AbortController();
    setStatus('loading');setError('');setParts([]);setView('front');
    let renderer:THREE.WebGLRenderer;
    try{renderer=new THREE.WebGLRenderer({antialias:true,powerPreference:'high-performance'});}
    catch{setStatus('error');setError('当前浏览器无法显示 3D。可以切换到视频对照继续查看动作。');return;}
    const canvas=renderer.domElement;container.appendChild(canvas);
    canvas.dataset.viewer='anatomy-3d';canvas.tabIndex=0;canvas.setAttribute('role','img');
    canvas.setAttribute('aria-label','可旋转的骨骼与肌肉解剖模型，方向键旋转，加减键缩放，0 重置');
    renderer.setPixelRatio(Math.min(window.devicePixelRatio||1,1.75));
    renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.toneMapping=THREE.ACESFilmicToneMapping;renderer.toneMappingExposure=1.1;
    const scene=new THREE.Scene();scene.background=new THREE.Color('#fafaf8');
    const camera=new THREE.PerspectiveCamera(28,1,.01,30);
    const controls=new OrbitControls(camera,canvas);controls.enableDamping=true;controls.dampingFactor=.09;
    controls.enablePan=true;controls.screenSpacePanning=true;controls.minDistance=.35;controls.maxDistance=8;
    controls.minPolarAngle=.1;controls.maxPolarAngle=Math.PI-.1;controls.rotateSpeed=.65;
    scene.add(new THREE.HemisphereLight('#ffffff','#b0a69b',1.15));
    const key=new THREE.DirectionalLight('#fffaf1',2.6);key.position.set(-3,5,5);scene.add(key);
    const fill=new THREE.DirectionalLight('#ffffff',.7);fill.position.set(3,2,1);scene.add(fill);
    const rim=new THREE.DirectionalLight('#ffffff',1.3);rim.position.set(0,3,-4);scene.add(rim);
    const meshes:PartMesh[]=[],geometries=new Set<THREE.BufferGeometry>(),materials=new Set<THREE.Material>();
    const center=new THREE.Vector3(0,.85,0),bounds=new THREE.Box3();let bodyHeight=1.7,bodyWidth=.7;

    const fitDistance=()=>Math.max(bodyHeight/2/Math.tan(THREE.MathUtils.degToRad(camera.fov/2)),bodyWidth/2/(Math.tan(THREE.MathUtils.degToRad(camera.fov/2))*camera.aspect))*1.22;
    const preset=(next:View)=>{
      const d=fitDistance();controls.target.copy(center);
      camera.position.copy(center).add(next==='front'?new THREE.Vector3(0,0,d):next==='back'?new THREE.Vector3(0,0,-d):new THREE.Vector3(d,0,0));
      controls.update();
    };
    const update=()=>{
      const state=latest.current;let visibleMuscles=0;
      for(const {mesh,part,material} of meshes){
        const regionMatch=state.focus==='all'||part.region===state.focus;
        const chosen=state.selected?.id===part.id;
        if(part.kind==='bone')mesh.visible=true;
        else if(state.layer==='skeleton')mesh.visible=false;
        else if(state.focus!=='all')mesh.visible=part.kind==='muscle'&&regionMatch;
        else mesh.visible=state.layer==='muscles'?true:part.referenceVisible;
        if(part.kind==='muscle'&&mesh.visible)visibleMuscles++;
        const color=part.kind==='bone'?'#d4cba0':part.kind==='connective'?'#ded5be':chosen?'#b64f42':'#cf7c74';
        material.color.set(color);material.emissive.set(chosen?'#70251d':'#000000');material.emissiveIntensity=chosen?.1:0;
        material.opacity=part.kind==='bone'?1:state.opacity;
        material.transparent=material.opacity<1;material.depthWrite=material.opacity>=.98;
      }
      canvas.dataset.layer=state.layer;canvas.dataset.focus=state.focus;canvas.dataset.selected=state.selected?.id??'';
      canvas.dataset.visibleMuscles=String(visibleMuscles);
    };
    const zoom=(factor:number)=>{const offset=camera.position.clone().sub(controls.target);offset.setLength(THREE.MathUtils.clamp(offset.length()*factor,controls.minDistance,controls.maxDistance));camera.position.copy(controls.target).add(offset);controls.update();};
    const focusPart=()=>{
      const state=latest.current,box=new THREE.Box3();
      // Hidden muscles still locate the corresponding bones in skeleton mode.
      for(const {mesh,part} of meshes)if(state.selected?part.id===state.selected.id:state.focus!=='all'&&part.region===state.focus)box.expandByObject(mesh);
      if(box.isEmpty())return;
      const target=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3());
      const distance=THREE.MathUtils.clamp(Math.max(size.y,size.x/camera.aspect,.25)/2/Math.tan(THREE.MathUtils.degToRad(camera.fov/2))*1.5,.45,7);
      const direction=camera.position.clone().sub(controls.target).normalize();controls.target.copy(target);camera.position.copy(target).addScaledVector(direction,distance);controls.update();
    };
    runtime.current={update,view:preset,zoom,focus:focusPart};
    const resize=()=>{const rect=container.getBoundingClientRect();if(!rect.width||!rect.height)return;const previous=camera.aspect;camera.aspect=rect.width/rect.height;camera.updateProjectionMatrix();renderer.setSize(rect.width,rect.height,false);if(Math.abs(previous-camera.aspect)>.15){preset('front');setView('front');}};
    const observer=new ResizeObserver(resize);observer.observe(container);resize();preset('front');
    const animate=()=>{if(disposed||lost)return;controls.update();renderer.render(scene,camera);raf=requestAnimationFrame(animate);};animate();
    const contextLost=(e:Event)=>{e.preventDefault();lost=true;cancelAnimationFrame(raf);setStatus('error');setError('3D 图形连接中断，请重新加载模型，或切换到视频对照。');};
    canvas.addEventListener('webglcontextlost',contextLost);
    const raycaster=new THREE.Raycaster(),pointer=new THREE.Vector2();let down:{x:number;y:number}|null=null;
    const pointerDown=(e:PointerEvent)=>{down={x:e.clientX,y:e.clientY};};
    const pointerUp=(e:PointerEvent)=>{
      if(!down||Math.hypot(e.clientX-down.x,e.clientY-down.y)>5){down=null;return;}down=null;
      const rect=canvas.getBoundingClientRect();pointer.set((e.clientX-rect.left)/rect.width*2-1,-(e.clientY-rect.top)/rect.height*2+1);raycaster.setFromCamera(pointer,camera);
      const hit=raycaster.intersectObjects(meshes.filter(m=>m.mesh.visible&&m.material.opacity>.05).map(m=>m.mesh),false)[0];
      if(!hit)return;const item=meshes.find(m=>m.mesh===hit.object);if(!item)return;
      setSelected(item.part);latest.current.onFocus(item.part.region??'all');
    };
    canvas.addEventListener('pointerdown',pointerDown);canvas.addEventListener('pointerup',pointerUp);
    const keydown=(e:KeyboardEvent)=>{
      if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','+','-','=','0'].includes(e.key))return;e.preventDefault();
      if(e.key==='0'){preset('front');setView('front');return;}if(e.key==='+'||e.key==='='){zoom(.85);return;}if(e.key==='-'){zoom(1.15);return;}
      const sphere=new THREE.Spherical().setFromVector3(camera.position.clone().sub(controls.target));
      sphere.theta+=(e.key==='ArrowLeft'?-.12:e.key==='ArrowRight'?.12:0);sphere.phi=THREE.MathUtils.clamp(sphere.phi+(e.key==='ArrowUp'?-.1:e.key==='ArrowDown'?.1:0),.1,Math.PI-.1);
      camera.position.copy(controls.target).add(new THREE.Vector3().setFromSpherical(sphere));controls.update();
    };canvas.addEventListener('keydown',keydown);

    // The model is hosted with the app. Fetch is abortable when switching views;
    // parsed GPU resources are also released if parsing finishes after unmount.
    fetch(MODEL_URL,{signal:abort.signal}).then(response=>{if(!response.ok)throw new Error('model unavailable');return response.arrayBuffer();})
      .then(buffer=>{if(disposed)return null;return new GLTFLoader().parseAsync(buffer,'');})
      .then(gltf=>{
        if(!gltf)return;
        gltf.scene.traverse(object=>{
          if(!(object instanceof THREE.Mesh))return;
          geometries.add(object.geometry);
          const original=Array.isArray(object.material)?object.material:[object.material];original.forEach(m=>m.dispose());
          const material=new THREE.MeshStandardMaterial({roughness:.78,metalness:0,side:THREE.DoubleSide});materials.add(material);object.material=material;
          meshes.push({mesh:object,part:describeAnatomyPart(object.userData,object.name),material});
        });
        if(disposed){geometries.forEach(g=>g.dispose());materials.forEach(m=>m.dispose());return;}
        if(meshes.length===0)throw new Error('empty model');
        scene.add(gltf.scene);bounds.setFromObject(gltf.scene);bounds.getCenter(center);
        const size=bounds.getSize(new THREE.Vector3());bodyHeight=size.y;bodyWidth=size.x;
        update();preset('front');setView('front');setParts(meshes.map(m=>m.part).filter(p=>p.kind==='muscle'&&p.layer!=='context'));canvas.dataset.meshCount=String(meshes.length);
        if(!lost)setStatus('ready');
      }).catch(cause=>{if(disposed||cause?.name==='AbortError')return;setStatus('error');setError('解剖模型加载失败，请重试。动作示意和视频对照仍可使用。');});

    return()=>{disposed=true;abort.abort();cancelAnimationFrame(raf);observer.disconnect();controls.dispose();runtime.current=null;
      canvas.removeEventListener('webglcontextlost',contextLost);canvas.removeEventListener('pointerdown',pointerDown);canvas.removeEventListener('pointerup',pointerUp);canvas.removeEventListener('keydown',keydown);
      geometries.forEach(g=>g.dispose());materials.forEach(m=>m.dispose());renderer.dispose();renderer.forceContextLoss();canvas.remove();};
  },[attempt]);

  function chooseView(next:View){setView(next);runtime.current?.view(next);}
  function chooseRegion(next:Focus){setSelected(null);onFocus(next);}
  function choosePart(id:string){const part=parts.find(p=>p.id===id);if(!part){setSelected(null);return;}setSelected(part);setLayer('muscles');onFocus(part.region??'all');}
  const copy=focus==='all'?null:REGION_ANATOMY_COPY[focus];
  return <section className="anatomy-card" aria-label="3D 骨骼与肌肉解剖视图">
    <header className="anatomy-heading"><span><Icon name="bone" size={18}/><strong>骨骼与肌肉</strong></span><span>ANATOMY / 3D</span></header>
    <div className="anatomy-layers" aria-label="解剖显示层">{(['reference','muscles','skeleton'] as Layer[]).map(value=><button key={value} className={layer===value?'active':''} aria-pressed={layer===value} onClick={()=>{setLayer(value);setSelected(null);}}>{{reference:'骨骼 + 主要肌群',muscles:'全部肌肉',skeleton:'仅骨骼'}[value]}</button>)}</div>
    <div className="anatomy-stage"><div className="anatomy-canvas" ref={host}/>
      <div className="anatomy-stage-label"><i/>解剖参考模型</div><div className="anatomy-tissue-key"><span><i/>骨骼</span>{layer!=='skeleton'&&<span><i/>肌肉</span>}</div>
      <div className="anatomy-presets" aria-label="解剖观察方向">{(['front','side','back'] as View[]).map(value=><button key={value} aria-pressed={view===value} onClick={()=>chooseView(value)} disabled={status!=='ready'}>{{front:'正面',side:'侧面',back:'背面'}[value]}</button>)}</div>
      <div className="anatomy-tools"><button aria-label="放大解剖模型" disabled={status!=='ready'} onClick={()=>runtime.current?.zoom(.8)}>+</button><button aria-label="缩小解剖模型" disabled={status!=='ready'} onClick={()=>runtime.current?.zoom(1.25)}>−</button><button aria-label="重置解剖视角" disabled={status!=='ready'} onClick={()=>chooseView('front')}><Icon name="refresh" size={15}/></button></div>
      <div className="anatomy-stage-hint">拖动旋转 · 双指缩放 · 点击肌肉查看名称</div>
      {status==='loading'&&<div className="anatomy-loading" role="status"><span className="spinner"/><strong>正在加载骨骼与肌肉…</strong><span>首次加载约 2.5 MB</span></div>}
      {status==='error'&&<div className="anatomy-error" role="alert"><Icon name="info" size={25}/><strong>解剖视图暂不可用</strong><p>{error}</p><button onClick={()=>setAttempt(n=>n+1)}>重新加载</button></div>}
    </div>
    <div className="anatomy-adjust"><label>肌肉透明度<input type="range" min={15} max={100} value={opacity*100} aria-label="肌肉透明度" disabled={layer==='skeleton'} onChange={e=>setOpacity(Number(e.target.value)/100)}/><span>{Math.round(opacity*100)}%</span></label><button disabled={status!=='ready'||(!selected&&focus==='all')} onClick={()=>runtime.current?.focus()}><Icon name="target" size={14}/>聚焦所选</button></div>
    <div className="anatomy-regions" aria-label="解剖身体区域"><button className={focus==='all'?'active':''} aria-pressed={focus==='all'} onClick={()=>chooseRegion('all')}>全身</button>{(Object.keys(regionNames) as BodyRegion[]).map(id=><button key={id} style={{'--region-color':REGION_COLORS[id]} as CSSProperties} className={focus===id?'active':''} aria-pressed={focus===id} onClick={()=>chooseRegion(focus===id?'all':id)}>{regionNames[id]}</button>)}</div>
    <div className="anatomy-selection"><div><span className="anatomy-selection-label">{selected?'已选择结构':copy?'部位解读':'认识身体结构'}</span><h4>{selected?.label??copy?.title??'从骨骼看支撑，从肌肉看连接'}</h4><p>{selected?selected.englishLabel:copy?.description??'选择一个部位，查看肌肉怎样跨过关节、连接骨骼。切换到「动作示意」可对照这次挥拍。'}</p>{copy&&<small>{copy.muscles}</small>}</div><select aria-label="选择解剖肌肉" value={selected?.kind==='muscle'?selected.id:''} disabled={status!=='ready'} onChange={e=>choosePart(e.target.value)}><option value="">查找肌肉…</option>{parts.filter(p=>focus==='all'||p.region===focus).map(p=><option value={p.id} key={p.id}>{p.label}</option>)}</select></div>
    <footer className="anatomy-footer"><p>标准解剖结构，用于理解身体部位；颜色不表示本次挥拍的肌肉激活或受力。</p><details><summary>模型来源与许可</summary><p>模型：<a href="https://github.com/slfresh/fitmitwith-anatomy-atlas" target="_blank" rel="noreferrer">Fit Mit With anatomy atlas</a>，源自 Z-Anatomy / BodyParts3D，CC BY-SA 4.0。本站调整显示颜色与可见层，几何模型保持原样。<a href="/models/anatomy/ATTRIBUTION.txt" target="_blank" rel="noreferrer">完整署名</a> · <a href="https://creativecommons.org/licenses/by-sa/4.0/" target="_blank" rel="noreferrer">许可证</a></p></details></footer>
  </section>;
}
