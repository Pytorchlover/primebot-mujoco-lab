"""Q1 actuator and action-chunk experiments. No VLA weights or hardware required."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import json, platform, time, xml.etree.ElementTree as ET
import mujoco
import numpy as np
from PIL import Image
import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'; OUT.mkdir(exist_ok=True)
SRC=ROOT/'sources/q1-urdf/xml/q1_v3_mc.xml'

def model(fixed=True):
    tree=ET.parse(SRC); root=tree.getroot()
    root.find('compiler').set('meshdir',str(SRC.parent.parent/'meshes'))
    if fixed:
        pelvis=root.find(".//body[@name='pelvis']")
        pelvis.remove(pelvis.find('freejoint'))
        # Lift feet clear of the ground: this is a suspended bench, not a balance controller.
        pelvis.set('pos','0 0 0.55')
        # Isolate actuator semantics from self-contact in this suspended bench.
        for geom in pelvis.iter('geom'):
            geom.set('contype','0'); geom.set('conaffinity','0')
        wb=root.find('worldbody')
        ET.SubElement(wb,'geom',name='fixture',type='box',pos='-0.13 0 0.27',size='.035 .06 .27',rgba='.12 .24 .34 1',contype='0',conaffinity='0')
    m=mujoco.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))
    return m

def mapping(m):
    assert np.all(m.actuator_trntype==mujoco.mjtTrn.mjTRN_JOINT)
    assert np.allclose(m.actuator_gear[:,0],1)
    j=m.actuator_trnid[:,0]
    assert np.all(m.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE)
    return m.jnt_qposadr[j],m.jnt_dofadr[j],m.jnt_range[j], [m.joint(int(x)).name for x in j]

def camera():
    c=mujoco.MjvCamera(); c.lookat[:]=[0,0,.55]; c.distance=1.75;c.azimuth=135;c.elevation=-14
    return c

def render(r,m,d,cam=None):
    mujoco.mj_forward(m,d)
    opt=mujoco.MjvOption(); opt.geomgroup[0]=0; opt.geomgroup[1]=1
    # Hide collision approximations, keep the floor and fixture visible in group 1.
    for name in ['floor','fixture']:
        idx=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,name)
        if idx>=0: m.geom_group[idx]=1
    r.update_scene(d,camera=cam or camera(),scene_option=opt)
    return r.render().copy()

def desired(t,base,idx,amp=.4):
    v=base.copy(); v[idx]=-.55+amp*np.sin(2*np.pi*.4*t); return v

def run(mode,amp=.4,H=10,visual=False):
    m=model(); d=mujoco.MjData(m); qa,va,limits,names=mapping(m)
    idx=names.index('left_shoulder_pitch_joint')
    base=np.zeros(m.nu); base[idx]=-.55
    d.qpos[qa]=base; mujoco.mj_forward(m,d)
    dt=float(m.opt.timestep); nsub=round(.05/dt); assert np.isclose(nsub*dt,.05)
    rows=[]; frames=[]; rgbs=[]; states=[]; actions=[]; next_states=[]; clips=0;sat=0
    r=mujoco.Renderer(m,height=540,width=960) if visual else None
    anchor=None; chunk=None; previous=None
    for k in range(120):
        t=float(d.time); target=desired(t,base,idx,amp)
        if k%H==0:
            anchor=d.qpos[qa].copy()
            chunk=np.stack([desired(t+h*.05,base,idx,amp)-anchor for h in range(H)])
            previous=anchor.copy()
        if mode=='accumulate':
            raw=previous+chunk[k%H]; previous=raw.copy()
        elif mode=='moving_anchor': raw=d.qpos[qa]+chunk[k%H]
        else: raw=anchor+chunk[k%H]
        command=np.clip(raw,limits[:,0],limits[:,1]); clips+=int(np.any(raw!=command))
        state=d.qpos[qa].copy()
        if visual:
            rgb=render(r,m,d); frames.append(rgb); rgbs.append(rgb[::3,::3].copy())
        states.append(state);actions.append(command.copy())
        rows.append([t,target[idx],command[idx],state[idx]])
        for _ in range(nsub):
            if mode=='angle_as_torque': tau=command.copy()
            else: tau=15*(command-d.qpos[qa])-.5*d.qvel[va]
            sat+=int(np.any((tau<m.actuator_ctrlrange[:,0])|(tau>m.actuator_ctrlrange[:,1])))
            d.ctrl[:]=np.clip(tau,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);mujoco.mj_step(m,d)
        next_states.append(d.qpos[qa].copy())
    a=np.asarray(rows); assert np.isfinite(d.qpos).all(); assert not any(w.number for w in d.warning)
    metric={'mode':mode,'amplitude_rad':amp,'H':H,'samples':120,'duration_s':float(d.time),'tracking_rmse_deg':float(np.rad2deg(np.sqrt(np.mean((a[:,1]-a[:,3])**2)))),'decode_rmse_deg':float(np.rad2deg(np.sqrt(np.mean((a[:,1]-a[:,2])**2)))),'clipped_control_samples':clips,'saturated_physics_steps':sat,'physics_steps':120*nsub}
    stem=f'{mode}_a{amp}_H{H}'
    np.savetxt(OUT/f'{stem}.csv',a,delimiter=',',header='time_s,target_rad,command_rad,observed_rad',comments='')
    if visual:
        r.close()
        imageio.mimsave(OUT/f'{mode}.mp4',frames,fps=20,macro_block_size=1)
        for k in [0,15,45,75]: Image.fromarray(frames[k]).save(OUT/f'{mode}_{k:03d}.png')
        if mode=='correct':
            np.savez_compressed(OUT/'episode.npz',time=a[:,0],image=np.asarray(rgbs),state=np.asarray(states),action=np.asarray(actions),next_state=np.asarray(next_states))
            (OUT/'episode.schema.json').write_text(json.dumps({'instruction':'Move the left shoulder periodically (scripted test)','policy':'analytic sine trajectory, NOT a VLA','image_shape':[180,320,3],'image':'external fixed camera, RGB uint8, before action','state':'actuated joint position at t, rad','action':'absolute joint target applied during [t,t+0.05), rad','next_state':'joint position after 50 physics steps','joint_names':names,'control_hz':20,'physics_hz':1000,'chunk_horizon':H,'base':'fixed, pelvis z=0.55m; robot collisions disabled; gravity enabled','delta_convention':'all H actions relative to the same chunk-start observed state'},indent=2))
    return metric,a

def main():
    start=time.perf_counter(); metrics=[]; traces={}
    for mode in ['correct','angle_as_torque','moving_anchor','accumulate']:
        metric,a=run(mode,visual=True);metrics.append(metric);traces[mode]=a
        print(metric,flush=True)
    for amp in [.2,.6]:
        for mode in ['correct','moving_anchor','accumulate']:
            metric,_=run(mode,amp=amp);metrics.append(metric)
    for H in [1,5,20]:
        for mode in ['correct','moving_anchor','accumulate']:
            metric,_=run(mode,H=H);metrics.append(metric)
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f7f9fc','axes.facecolor':'#f7f9fc'})
    fig,ax=plt.subplots(figsize=(11,4.5)); a=traces['correct'];ax.plot(a[:,0],np.rad2deg(a[:,1]),'--',color='#172b4d',label='Target')
    for mode,label,color in [('correct','PD (Kp=15, Kd=0.5)','#087e8b'),('angle_as_torque','Angle written into motor.ctrl','#e16b42')]:
        a=traces[mode];ax.plot(a[:,0],np.rad2deg(a[:,3]),label=label,color=color)
    ax.set(xlabel='Simulation time (s)',ylabel='Left shoulder pitch (deg)',title='Same target, different actuator semantics');ax.legend();fig.tight_layout();fig.savefig(OUT/'pd_comparison.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(11,7),sharex=True)
    for mode,label,color in [('correct','Fixed anchor (correct)','#087e8b'),('moving_anchor','Moving anchor (wrong)','#d28e00'),('accumulate','Cumulative sum (wrong)','#b84c65')]:
        a=traces[mode];axs[0].plot(a[:,0],np.rad2deg(a[:,2]),label=label,color=color);axs[1].plot(a[:,0],np.rad2deg(a[:,3]),color=color)
    for ax in axs:
        ax.plot(a[:,0],np.rad2deg(a[:,1]),'--',color='#172b4d',label='Target');ax.set_ylabel('Shoulder pitch (deg)')
    axs[0].set_title('Identical delta convention, three decoders (H=10, 20 Hz)');axs[0].legend(ncol=2);axs[1].set(xlabel='Simulation time (s)',title='Observed physical response');fig.tight_layout();fig.savefig(OUT/'delta_comparison.png',dpi=180);plt.close(fig)
    # Original floating-base, zero-input experiment.
    m=model(False); d=mujoco.MjData(m); rr=mujoco.Renderer(m,height=540,width=960); drops=[];dropframes=[]
    for k in range(2001):
        if k in [0,250,750,2000]:
            drops.append({'time':float(d.time),'pelvis_z':float(d.qpos[2]),'contacts':int(d.ncon)})
            im=render(rr,m,d);dropframes.append(im);Image.fromarray(im).save(OUT/f'free_{k:04d}.png')
        if k<2000:mujoco.mj_step(m,d)
    rr.close()
    (OUT/'free_drop.json').write_text(json.dumps(drops,indent=2))
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))
    (OUT/'environment.json').write_text(json.dumps({'python':platform.python_version(),'mujoco':mujoco.__version__,'numpy':np.__version__,'platform':platform.platform(),'elapsed_s':time.perf_counter()-start},indent=2))
    # Load and render the unmodified T1 terrain as an overview, no locomotion claim.
    m=mujoco.MjModel.from_xml_path(str(ROOT/'sources/t1-urdf/xml/scene_terrain.xml'));d=mujoco.MjData(m);m.vis.global_.offwidth=1440;m.vis.global_.offheight=900
    r=mujoco.Renderer(m,height=900,width=1440);c=camera();c.lookat[:]=[2,2,.2];c.distance=12;c.azimuth=120;c.elevation=-48
    mujoco.mj_forward(m,d);r.update_scene(d,camera=c);Image.fromarray(r.render()).save(OUT/'t1_terrain.png');r.close()
    print('DONE',flush=True)
if __name__=='__main__': main()
