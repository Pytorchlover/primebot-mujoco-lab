"""Validate saved evidence and replay recorded targets through the same physics."""
import json
import numpy as np
import mujoco
from experiment import OUT, model, mapping
z=np.load(OUT/'episode.npz');schema=json.loads((OUT/'episode.schema.json').read_text())
assert z['image'].shape==(120,180,320,3)
assert z['image'].dtype==np.uint8
for key in ['state','action','next_state']: assert z[key].shape==(120,22) and np.isfinite(z[key]).all()
assert np.allclose(np.diff(z['time']),.05)
assert np.allclose(z['next_state'][:-1],z['state'][1:],atol=1e-10)
m=model();d=mujoco.MjData(m);qa,va,limits,names=mapping(m)
assert names==schema['joint_names']
d.qpos[qa]=z['state'][0];mujoco.mj_forward(m,d)
errors=[]
for k,action in enumerate(z['action']):
 errors.append(float(np.max(np.abs(d.qpos[qa]-z['state'][k]))))
 for _ in range(50):
  tau=15*(action-d.qpos[qa])-.5*d.qvel[va]
  d.ctrl[:]=np.clip(tau,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]);mujoco.mj_step(m,d)
assert max(errors)<1e-10
metrics=json.loads((OUT/'metrics.json').read_text()); h1=[x for x in metrics if x['H']==1]
assert len(h1)==3
assert max(x['tracking_rmse_deg'] for x in h1)-min(x['tracking_rmse_deg'] for x in h1)<1e-9
assert all(x['saturated_physics_steps']==0 and x['clipped_control_samples']==0 for x in metrics if x['mode']=='correct')
report={'checks':'PASS','episode_samples':120,'physics_steps':6000,'replay_max_abs_error_rad':max(errors),'H1_negative_control':'all three decoders identical','experiment_runs':len(metrics)}
(OUT/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
