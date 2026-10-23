"""Collect synchronized RGB/state/action episodes and audit action + image replay."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
from pathlib import Path
import json, hashlib, time, xml.etree.ElementTree as ET
import numpy as np
import mujoco
from PIL import Image
import imageio.v2 as imageio
from experiment import ROOT, SRC, mapping
OUT=ROOT/'multiview'; OUT.mkdir(exist_ok=True)
W,H=480,360
CAMS={'front':([1.5,0,1.0],[0,0,.65]),'side':([0,1.5,1.0],[0,0,.65]),'overhead':([0,0,2.0],[0,0,.55])}
TASKS={'left_reach':'Raise and lower the left arm','right_reach':'Raise and lower the right arm','both_turn':'Raise both arms while turning the waist'}
STATE_SPEC=mujoco.mjtState.mjSTATE_INTEGRATION

def build():
    root=ET.parse(SRC).getroot();root.find('compiler').set('meshdir',str(SRC.parent.parent/'meshes'))
    pelvis=root.find(".//body[@name='pelvis']");pelvis.remove(pelvis.find('freejoint'));pelvis.set('pos','0 0 0.55')
    for g in pelvis.iter('geom'):g.set('contype','0');g.set('conaffinity','0')
    wb=root.find('worldbody')
    ET.SubElement(wb,'geom',name='fixture',type='box',pos='-.13 0 .27',size='.035 .06 .27',rgba='.15 .27 .35 1',contype='0',conaffinity='0',group='1')
    # Camera -Z points at the target, +Y points up.
    for name,(pos,target) in CAMS.items():
        z=np.array(pos)-target;z=z/np.linalg.norm(z)
        up=np.array([0.,1.,0.]) if name=='overhead' else np.array([0.,0.,1.])
        x=np.cross(up,z);x/=np.linalg.norm(x);y=np.cross(z,x)
        ET.SubElement(wb,'camera',name=name,pos=' '.join(map(str,pos)),xyaxes=' '.join(map(str,np.r_[x,y])),fovy='45')
    m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    m.geom_group[m.geom('floor').id]=1
    return m

def draw(renderer,d,name):
    # Caller freezes physics and refreshes forward kinematics before all cameras.
    opt=mujoco.MjvOption();opt.geomgroup[0]=0;opt.geomgroup[1]=1
    renderer.update_scene(d,camera=name,scene_option=opt)
    return renderer.render().copy()

def command(task,k,names):
    # Smooth 4-second reach-return motion. Exactly one target per control tick.
    phase=np.sin(np.pi*k/80)**2
    q=np.zeros(len(names))
    arms=['left'] if task=='left_reach' else ['right'] if task=='right_reach' else ['left','right']
    for arm in arms:
        q[names.index(arm+'_shoulder_pitch_joint')]=-1.15*phase
        q[names.index(arm+'_elbow_joint')]=.75*phase
    if task=='both_turn':q[names.index('waist_yaw_joint')]=.45*np.sin(2*np.pi*k/80)*phase
    return q

def step(m,d,q,qa,va):
    for _ in range(50):
        tau=15*(q-d.qpos[qa])-.5*d.qvel[va]
        d.ctrl[:]=np.clip(tau,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);mujoco.mj_step(m,d)

def getstate(m,d):
    s=np.empty(mujoco.mj_stateSize(m,STATE_SPEC));mujoco.mj_getState(m,d,s,STATE_SPEC);return s

def collect(task):
    ep=OUT/task;ep.mkdir(exist_ok=True)
    for name in CAMS:(ep/name).mkdir(exist_ok=True)
    m=build();d=mujoco.MjData(m);qa,va,limits,names=mapping(m);r=mujoco.Renderer(m,height=H,width=W)
    mujoco.mj_forward(m,d)
    initial=getstate(m,d);rows=[];records={k:[] for k in ['time','state','velocity','action','next_state','snapshot']}
    video=[]
    for k in range(80):
        t=float(d.time);mujoco.mj_forward(m,d);q=np.clip(command(task,k,names),limits[:,0],limits[:,1])
        records['time'].append(t);records['state'].append(d.qpos[qa].copy());records['velocity'].append(d.qvel[va].copy());records['action'].append(q.copy());records['snapshot'].append(getstate(m,d))
        images={};tiles=[]
        for cam in CAMS:
            rgb=draw(r,d,cam);path=ep/cam/f'{k:04d}.png';Image.fromarray(rgb).save(path)
            images[cam]={'file':str(path.relative_to(OUT)),'sim_time_s':float(d.time),'frame_id':k,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()};tiles.append(rgb)
        assert float(d.time)==t # Rendering did not advance physics.
        video.append(np.concatenate(tiles,axis=1))
        row={'episode':task,'frame_id':k,'sim_time_s':t,'instruction':TASKS[task],'images':images}
        step(m,d,q,qa,va);records['next_state'].append(d.qpos[qa].copy());rows.append(row)
    r.close();assert np.isfinite(d.qpos).all() and not any(w.number for w in d.warning)
    arrays={k:np.asarray(v) for k,v in records.items()};np.savez_compressed(ep/'trajectory.npz',**arrays,initial_integration_state=initial)
    (ep/'index.jsonl').write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
    imageio.mimsave(ep/'three_views.mp4',video,fps=20,macro_block_size=1)
    return rows

def audit_sync(rows):
    issues=[]
    for i,row in enumerate(rows):
        if row['frame_id']!=i or not np.isclose(row['sim_time_s'],i*.05,atol=1e-10):issues.append(i)
        for cam in CAMS:
            rec=row['images'][cam]
            if rec['frame_id']!=row['frame_id'] or not np.isclose(rec['sim_time_s'],row['sim_time_s'],atol=1e-10):issues.append(i)
    return sorted(set(issues))

def audit(task,rows):
    ep=OUT/task;z=np.load(ep/'trajectory.npz');m=build();d=mujoco.MjData(m);qa,va,limits,names=mapping(m)
    assert z['state'].shape==(80,22);assert not audit_sync(rows)
    assert np.allclose(z['next_state'][:-1],z['state'][1:],atol=1e-10)
    mujoco.mj_setState(m,d,z['initial_integration_state'],STATE_SPEC);mujoco.mj_forward(m,d)
    replay=[]
    for k,q in enumerate(z['action']):
        mujoco.mj_forward(m,d);replay.append(float(np.max(abs(d.qpos[qa]-z['state'][k]))));step(m,d,q,qa,va)
    assert max(replay)<1e-10
    r=mujoco.Renderer(m,height=H,width=W);pixel_max=0;pixel_maes=[];shifted_pixel_diffs=[]
    for k in range(80):
        mujoco.mj_resetData(m,d)
        mujoco.mj_setState(m,d,z['snapshot'][k],STATE_SPEC);mujoco.mj_forward(m,d)
        for cam in CAMS:
            rgb=draw(r,d,cam);path=OUT/rows[k]['images'][cam]['file']
            assert hashlib.sha256(path.read_bytes()).hexdigest()==rows[k]['images'][cam]['sha256']
            saved=np.asarray(Image.open(path));assert saved.shape==(H,W,3) and saved.dtype==np.uint8
            diff=abs(rgb.astype(int)-saved.astype(int));pixel_max=max(pixel_max,int(diff.max()));pixel_maes.append(float(diff.mean()))
            if cam=='side' and k<79:
                shifted=np.asarray(Image.open(OUT/rows[k+1]['images'][cam]['file']))
                shifted_pixel_diffs.append(float(np.mean(abs(rgb.astype(float)-shifted.astype(float)))))
        if k==25:
            Image.fromarray(draw(r,d,'front')).save(ep/'replayed_0025.png')
    r.close()
    # Report rasterization differences quantitatively; do not assert bitwise equality.
    # Fault injection changes index copies only, never corrupts the delivered dataset.
    shifted=json.loads(json.dumps(rows[:-1]))
    for k in range(79):shifted[k]['images']['side']=rows[k+1]['images']['side']
    detected=audit_sync(shifted);assert len(detected)==79
    dropped=json.loads(json.dumps(rows));del dropped[20];assert audit_sync(dropped)
    report={'samples':80,'views':3,'rgb_frames':240,'duration_s':4.0,'last_observation_s':float(z['time'][-1]),'sync_max_skew_s':0.0,'action_replay_max_error_rad':max(replay),'snapshot_render_max_pixel_error':pixel_max,'snapshot_render_mean_pixel_MAE':float(np.mean(pixel_maes)),'snapshot_render_worst_frame_MAE':max(pixel_maes),'side_shift_ms':50,'shifted_rows':79,'detected_shifted_rows':len(detected),'hidden_shift_mean_pixel_MAE':float(np.mean(shifted_pixel_diffs)),'dropped_row_detected':True}
    (ep/'validation.json').write_text(json.dumps(report,indent=2));return report

def main():
    start=time.perf_counter();allrows=[];reports={}
    m=build();d=mujoco.MjData(m);mujoco.mj_forward(m,d);_,_,_,names=mapping(m)
    cameras={}
    for cam in CAMS:
        i=m.camera(cam).id;cameras[cam]={'world_position_m':d.cam_xpos[i].tolist(),'camera_to_world_rotation':d.cam_xmat[i].reshape(3,3).tolist(),'fovy_deg':float(m.cam_fovy[i]),'width':W,'height':H,'axes':'MuJoCo/OpenGL: +X right, +Y up, looks along -Z'}
    (OUT/'schema.json').write_text(json.dumps({'joint_names':names,'units':'rad; rad/s','action':'absolute joint target held for 50 physics steps','timestamps':'simulation time before action, not wall clock','physics_dt_s':.001,'control_dt_s':.05,'cameras':cameras,'state_spec':'mjSTATE_INTEGRATION','mujoco':mujoco.__version__,'base':'pelvis fixed at z=0.55m; robot collisions disabled; gravity retained','policy':'three scripted motions; instruction is a task label, not language model input'},indent=2))
    for task in TASKS:
        rows=collect(task);report=audit(task,rows);allrows+=rows;reports[task]=report;print(task,report,flush=True)
    (OUT/'report.json').write_text(json.dumps({'episodes':reports,'total_samples':240,'total_rgb_frames':720,'total_simulation_s':12,'wall_elapsed_s':time.perf_counter()-start},indent=2))
    # Standalone file viewer: manifest is embedded, image paths are relative.
    page='''<!doctype html><meta charset="utf-8"><title>Q1 多视角数据检查</title><style>body{background:#14202c;color:#e4edf4;font:16px/1.6 sans-serif;margin:24px}header{display:flex;gap:20px;align-items:center}main{display:flex;gap:12px}figure{margin:16px 0;flex:1}img{width:100%}input{width:65%}pre{background:#213344;padding:15px}</style><h1>Q1 多视角数据检查</h1><header><select id="ep"></select><button id="play">播放</button><input id="frame" type="range" min="0" max="79" value="0"><b id="stamp"></b></header><main>'''
    for cam in CAMS:page+=f'<figure><figcaption>{cam}</figcaption><img id="{cam}"></figure>'
    page+='</main><pre id="info"></pre><script>const rows='+json.dumps(allrows)+'; const cams='+json.dumps(list(CAMS))+';'
    page+='''const ep=document.querySelector('#ep'),slider=document.querySelector('#frame');[...new Set(rows.map(x=>x.episode))].forEach(x=>ep.add(new Option(x,x)));function show(){const r=rows.find(x=>x.episode===ep.value&&x.frame_id===+slider.value);cams.forEach(c=>document.getElementById(c).src=r.images[c].file);document.querySelector('#stamp').textContent=`frame ${r.frame_id} / ${r.sim_time_s.toFixed(2)} s`;document.querySelector('#info').textContent=JSON.stringify(r,null,2)}ep.onchange=()=>{slider.value=0;show()};slider.oninput=show;let timer=null;document.querySelector('#play').onclick=()=>{if(timer){clearInterval(timer);timer=null;document.querySelector('#play').textContent='播放';return}document.querySelector('#play').textContent='暂停';timer=setInterval(()=>{slider.value=(+slider.value+1)%80;show()},50)};document.onkeydown=e=>{if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();slider.value=Math.max(0,Math.min(79,+slider.value+(e.key==='ArrowRight'?1:-1)));show()}};show();</script>'''
    (OUT/'viewer.html').write_text(page);print('DONE',flush=True)
if __name__=='__main__':main()
