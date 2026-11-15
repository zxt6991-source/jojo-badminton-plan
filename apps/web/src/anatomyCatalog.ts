import type {BodyRegion} from '@badminton/metrics';

export type AnatomyFocus=BodyRegion|'all';

export interface AnatomyPart {
  id:string;
  label:string;
  englishLabel:string;
  region:BodyRegion|null;
  layer:'superficial'|'deep'|'context';
  kind:'muscle'|'bone'|'connective';
  referenceVisible:boolean;
}

// Keys and layer assignments are the atlas's glTF extras. Display labels do not
// imply that a muscle's activation, force, or individual anatomy was measured.
const muscleLabels:Record<string,string>={
  deltoid_anterior:'三角肌前束',
  deltoid_lateral:'三角肌中束',
  deltoid_posterior:'三角肌后束',
  supraspinatus:'冈上肌',
  infraspinatus:'冈下肌',
  subscapularis:'肩胛下肌',
  teres_minor:'小圆肌',
  teres_major:'大圆肌',
  rectus_femoris:'股直肌',
  vastus_lateralis:'股外侧肌',
  vastus_medialis:'股内侧肌',
  vastus_intermedius:'股中间肌',
  biceps_femoris_long:'股二头肌长头',
  biceps_femoris_short:'股二头肌短头',
  semitendinosus:'半腱肌',
  semimembranosus:'半膜肌',
  sartorius:'缝匠肌',
  adductor_magnus:'大收肌',
  adductor_longus:'长收肌',
  adductor_brevis:'短收肌',
  gracilis:'股薄肌',
  pectineus:'耻骨肌',
  pectoralis_major_clavicular:'胸大肌锁骨部',
  pectoralis_major_sternocostal:'胸大肌胸肋部',
  pectoralis_major_abdominal:'胸大肌腹部',
  pectoralis_minor:'胸小肌',
  serratus_anterior:'前锯肌',
  trapezius_upper:'斜方肌上部',
  trapezius_middle:'斜方肌中部',
  trapezius_lower:'斜方肌下部',
  latissimus_dorsi:'背阔肌',
  rhomboid_major:'大菱形肌',
  rhomboid_minor:'小菱形肌',
  levator_scapulae:'肩胛提肌',
  serratus_posterior_inferior:'下后锯肌',
  serratus_posterior_superior:'上后锯肌',
  iliocostalis_lumborum:'髂肋肌腰部',
  iliocostalis_thoracis:'髂肋肌胸部',
  longissimus_thoracis:'胸最长肌',
  spinalis_thoracis:'胸棘肌',
  multifidus_lumborum:'腰多裂肌',
  rectus_abdominis:'腹直肌',
  transversus_abdominis:'腹横肌',
  external_oblique:'腹外斜肌',
  internal_oblique:'腹内斜肌',
  quadratus_lumborum:'腰方肌',
  sternocleidomastoid:'胸锁乳突肌',
  scalenus_medius:'中斜角肌',
  scalenus_anterior:'前斜角肌',
  scalenus_posterior:'后斜角肌',
  splenius_capitis:'头夹肌',
  splenius_colli:'颈夹肌',
  biceps_brachii_long:'肱二头肌长头',
  biceps_brachii_short:'肱二头肌短头',
  brachialis:'肱肌',
  coracobrachialis:'喙肱肌',
  triceps_long:'肱三头肌长头',
  triceps_lateral:'肱三头肌外侧头',
  triceps_medial:'肱三头肌内侧头',
  brachioradialis_muscle:'肱桡肌',
  flexor_carpi_radialis:'桡侧腕屈肌',
  palmaris_longus_muscle:'掌长肌',
  humeral_head_of_flexor_carpi_ulnaris:'尺侧腕屈肌肱骨头',
  ulnar_head_of_flexor_carpi_ulnaris:'尺侧腕屈肌尺骨头',
  humero_ulnar_head_of_flexor_digitorum_superficialis:'指浅屈肌肱尺头',
  radial_head_of_flexor_digitorum_superficialis:'指浅屈肌桡骨头',
  flexor_digitorum_profundus:'指深屈肌',
  flexor_pollicis_longus:'拇长屈肌',
  pronator_quadratus:'旋前方肌',
  superficial_head_of_pronator_teres:'旋前圆肌浅头',
  deep_head_of_pronator_teres:'旋前圆肌深头',
  supinator:'旋后肌',
  extensor_digitorum:'指伸肌',
  ulnar_head_of_extensor_carpi_ulnaris:'尺侧腕伸肌尺骨头',
  humeral_head_of_extensor_carpi_ulnaris:'尺侧腕伸肌肱骨头',
  extensor_carpi_radialis_longus:'桡侧腕长伸肌',
  extensor_carpi_radialis_brevis:'桡侧腕短伸肌',
  anconeus_muscle:'肘肌',
  extensor_digiti_minimi:'小指伸肌',
  abductor_pollicis_longus:'拇长展肌',
  extensor_indicis:'示指伸肌',
  extensor_pollicis_brevis:'拇短伸肌',
  extensor_pollicis_longus:'拇长伸肌',
  gluteus_maximus:'臀大肌',
  gluteus_medius:'臀中肌',
  gluteus_minimus:'臀小肌',
  iliacus:'髂肌',
  psoas_major:'腰大肌',
  tensor_fasciae_latae:'阔筋膜张肌',
  piriformis:'梨状肌',
  quadratus_femoris:'股方肌',
  obturator_internus:'闭孔内肌',
  obturator_externus:'闭孔外肌',
  gemellus_inferior:'下孖肌',
  gemellus_superior:'上孖肌',
  gastrocnemius_lateral:'腓肠肌外侧头',
  gastrocnemius_medial:'腓肠肌内侧头',
  soleus:'比目鱼肌',
  plantaris:'跖肌',
  tibialis_anterior:'胫骨前肌',
  tibialis_posterior:'胫骨后肌',
  fibularis_longus:'腓骨长肌',
  fibularis_brevis:'腓骨短肌',
  fibularis_tertius:'第三腓骨肌',
  extensor_digitorum_longus:'趾长伸肌',
  extensor_hallucis_longus:'拇趾长伸肌',
  flexor_digitorum_longus:'趾长屈肌',
  flexor_hallucis_longus:'拇趾长屈肌',
  popliteus:'腘肌',
};

const regionMap:Record<string,BodyRegion>={
  shoulder:'upperArm',arms:'upperArm',forearms:'forearm',
  hips:'legs',thigh:'legs',calves:'legs',
  chest:'trunk',back:'trunk',core:'trunk',neck:'trunk',
};

const text=(value:unknown)=>typeof value==='string'?value:'';

export function describeAnatomyPart(data:Record<string,unknown>,name:string):AnatomyPart {
  const englishLabel=text(data.label)||name;
  if(data.boneId==='skeleton'||data.contextType==='bone'){
    return {id:text(data.boneId)||name,label:'全身骨骼',englishLabel,region:null,layer:'context',kind:'bone',referenceVisible:true};
  }
  if(data.supportId==='connective_tissue'||data.contextType==='connective'){
    return {id:text(data.supportId)||name,label:'结缔组织',englishLabel,region:null,layer:'context',kind:'connective',referenceVisible:false};
  }
  if(data.supportId==='head_hands_feet'){
    return {id:'head_hands_feet',label:'头、手与足部肌肉',englishLabel,region:null,layer:'context',kind:'muscle',referenceVisible:false};
  }
  const key=text(data.key),sourceRegion=text(data.region);
  const region=regionMap[sourceRegion]??null;
  const side=data.side==='left'?'左侧':data.side==='right'?'右侧':'';
  const layer=data.layer==='deep'?'deep':data.layer==='superficial'?'superficial':'context';
  return {
    id:text(data.muscleId)||name,
    label:side+(muscleLabels[key]||englishLabel),
    englishLabel,
    region,
    layer,
    kind:'muscle',
    referenceVisible:layer==='superficial'&&['shoulder','arms','thigh','calves'].includes(sourceRegion),
  };
}

export const REGION_ANATOMY_COPY:Record<BodyRegion,{title:string;description:string;muscles:string}>={
  legs:{
    title:'下肢 · 髋、膝与踝',
    description:'从骨盆沿大腿、小腿看到足部，理解这些关节和周围肌肉的位置。再对照动作回放中的屈膝与伸膝变化。',
    muscles:'股四头肌、腘绳肌、臀肌与小腿肌群',
  },
  trunk:{
    title:'躯干 · 脊柱与胸廓',
    description:'脊柱、胸廓和骨盆构成躯干的骨性框架。观察腹部、背部肌肉如何分布，再回看画面中的躯干倾斜。',
    muscles:'腹直肌、腹斜肌、竖脊肌与背阔肌',
  },
  upperArm:{
    title:'肩与上臂 · 从肩到肘',
    description:'肩部肌肉围绕肩关节，肱二头肌和肱三头肌分布在上臂前后。转动模型看清位置，再对照肩到肘的方向变化。',
    muscles:'三角肌、肩袖肌群、肱二头肌与肱三头肌',
  },
  forearm:{
    title:'前臂 · 从肘到腕',
    description:'桡骨与尺骨之间和周围分布着多组肌肉与肌腱。可分层查看，再回到动作示意观察肘到腕的方向和手腕轨迹。',
    muscles:'前臂屈肌群、伸肌群、旋前与旋后肌群',
  },
};
