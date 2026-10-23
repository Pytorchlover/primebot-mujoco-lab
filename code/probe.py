from pathlib import Path
import json,mujoco
root=Path(__file__).resolve().parents[1]
rows=[]
for repo in ['q1-urdf','t1-urdf']:
 for p in sorted((root/'sources'/repo/'xml').glob('*.xml')):
  try:
   m=mujoco.MjModel.from_xml_path(str(p)); d=mujoco.MjData(m)
   for _ in range(100): mujoco.mj_step(m,d)
   rows.append(dict(file=str(p.relative_to(root)),nq=m.nq,nv=m.nv,nu=m.nu,dt=m.opt.timestep,time=d.time))
  except Exception as e: rows.append(dict(file=str(p.relative_to(root)),error=str(e)))
print(json.dumps(rows,indent=2))
