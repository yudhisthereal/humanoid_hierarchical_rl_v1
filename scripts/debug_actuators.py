import numpy as np
import mujoco
from pathlib import Path
from importlib import util

proj = Path(__file__).resolve().parents[1]
mod_path = proj / "scripts" / "brace_only_single" / "brace_only_single.py"
spec = util.spec_from_file_location("mod", mod_path)
mod = util.module_from_spec(spec)
spec.loader.exec_module(mod)

env = mod.HumanoidEnv()
print("Actuator names and ctrl ranges:")
for aid in env.actuator_ids:
    name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
    lo = float(env.model.actuator_ctrlrange[aid,0])
    hi = float(env.model.actuator_ctrlrange[aid,1])
    print(f"  {aid}: {name} range=({lo}, {hi})")

# Find arm/forearm actuator indices
def aid_for(name):
    a = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    return int(a) if a>=0 else None

pa = aid_for("pos_arm")
pf = aid_for("pos_forearm")
print("pos_arm id", pa, "pos_forearm id", pf)

print("Initial qpos arm/forearm:")
for jn in ["arm_joint","forearm_joint","arm_left_joint","forearm_left_joint"]:
    j = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
    if j>=0:
        print(" ", jn, env.data.qpos[int(env.model.jnt_qposadr[j])])

print("Setting desired ctrl targets and stepping once.")
# set explicit action in actuator units (not normalized)
action = np.zeros((env.action_dim,), dtype=np.float32)
# fill actions with ctrl targets; find actuator order in env.actuator_ids
for i, aid in enumerate(env.actuator_ids):
    name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
    if name == "pos_arm":
        action[i] = 0.129
    if name == "pos_forearm":
        action[i] = 6.0

print("Action values to apply:", action)
# step once and show qpos before/after
print("qpos before:", env.data.qpos.copy())
obs, r, done, info = env.step(action)
print("qpos after step:", env.data.qpos.copy())
print("data.ctrl (first 10):", env.data.ctrl[:10])
print("qvel (arm/forearm):")
for jn in ["arm_joint","forearm_joint","arm_left_joint","forearm_left_joint"]:
    j = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, jn)
    if j>=0:
        print(" ", jn, env.data.qvel[int(env.model.jnt_dofadr[j])])