# Bimanual Handover — Isaac Lab 3.0 port

Ported from the `aurova_reinforcement_learning/bimanual_handover` task in the old
`omni.isaac.lab_tasks` (Isaac Lab 1.2.0) repo. Target: Isaac Lab 3.0 (beta2),
**PhysX backend**, installed **kit-less** via the `ov[ovphysx]` runtime wheel so it
runs on Capella's H100 nodes without needing a full Isaac Sim / Omniverse Kit
install (no Vulkan/RTX driver dependency — see the "why" section below).

This directory is a standalone, pip-installable task package — it does **not**
contain Isaac Lab itself. Isaac Lab 3.0 gets cloned separately on the cluster.

## Why kit-less `ov[ovphysx]`, not full Isaac Sim

Isaac Sim's Kit app boots a Vulkan/EGL-based renderer at startup even for
headless, no-camera RL training. Capella's H100 nodes are provisioned as
compute-only (no GL/Vulkan driver stack for a display-less datacenter GPU),
so Kit was silently falling back to Mesa's `llvmpipe` software rasterizer —
which is the real cause of the `LLVM: out of memory` crash, independent of
`--mem`/GPU exclusivity.

Isaac Lab 3.0 splits physics from rendering into separate installable
packages. `ov[ovphysx]` is a standalone PhysX runtime wheel that needs no Kit,
no Vulkan, no display driver — pure CUDA. Same `isaaclab_physx` Python API as
full Isaac Sim, so this task's code doesn't change based on which one you
install; only the cluster-side install command differs.

Not used here: the Newton/Warp backend. Isaac Lab's Newton docs currently only
confirm classic-RL / flat-terrain-locomotion coverage — this task is
contact-rich dual-arm + underactuated-hand manipulation (17 contact sensors,
custom force-based rewards), so it stays on PhysX.

## What changed in the port

- `omni.isaac.lab*` → `isaaclab*` / `isaaclab_physx` imports (Isaac Lab 3.0
  renamed the extension namespace).
- Asset `.data.*` reads (`joint_pos`, `body_state_w`, `root_state_w`, etc.)
  now return a `ProxyArray`; `.torch` is appended at every read site to get
  an explicit `torch.Tensor` (there's also a transparent bridge that works
  without this, but it spams `DeprecationWarning`).
- `Articulation.write_joint_state_to_sim(pos, vel, joint_ids, env_ids)` was
  **removed** (not just deprecated — raises `NotImplementedError` in 3.0).
  Split into `write_joint_position_to_sim_index(...)` +
  `write_joint_velocity_to_sim_index(...)` (`reset_robot`, `reset_robot_ee`
  in `bimanual_direct_env.py`).
- `set_joint_position_target` → `set_joint_position_target_index` and
  `write_root_pose_to_sim` → `write_root_pose_to_sim_index` (both had
  working deprecated aliases, but renamed anyway to match the current
  recommended API and avoid warning spam under 4096+ envs).
- `SimulationCfg` now takes an explicit `physics=` backend config
  (`PhysxCfg()` from `isaaclab_physx.physics`) instead of an implicit
  PhysX default.
- Fixed a hardcoded absolute USD path
  (`/home/hapi039h/isaaclab/...`, a leftover from a previous cluster user)
  in `robots_cfg.py` — now resolved relative to the package, via
  `assets/config/usd/`.
- Fixed a pre-existing bug in `mdp/__init__.py`: `from utils import *` (missing
  the relative dot — would have failed at import) → `from .utils import *`.

## What was *not* changed (left as deprecated-but-working, or unverified)

- `root_physx_view.get_jacobians()` in `_get_ee_pose()` — the property is
  deprecated in favor of `root_view` / `data.body_link_jacobian_w`, but still
  works today. Left as-is because I could not verify the new accessor's
  indexing convention without a live GPU run. **Validate this first** if you
  hit jacobian-shape errors.
- `ContactSensor.data.force_matrix_w` — assumed unchanged; not explicitly
  covered in the migration notes I could check.
- Camera path (`self.scene["camera"]`, `save_images_grid`) — `render_imgs`
  defaults to `False` in `BimanualDirectCfg`, so this code path is inactive
  by default. If you turn it on, note there's no `camera` sensor actually
  registered in `_setup_scene()` in the original code either — that looks
  like a pre-existing latent bug, not something the migration introduced.

None of this was executed — there's no Linux/NVIDIA GPU available in the
environment this port was written in. Treat this as a careful, systematic
translation against Isaac Lab 3.0's migration guide and source, not a
tested one. Budget time for a first real run to shake out anything missed.

## Setup on Capella

### 1. Clone Isaac Lab 3.0 and install kit-less (on a Capella login/build node)

```bash
git clone -b release/3.0.0-beta2 https://github.com/isaac-sim/IsaacLab.git ~/isaaclab3
cd ~/isaaclab3
python3 -m venv ~/isaaclab3-venv
source ~/isaaclab3-venv/bin/activate

# Kit-less install: PhysX via the ovphysx runtime wheel + SB3 only (this task
# only ships an sb3_ppo_cfg.yaml). No Isaac Sim, no Vulkan/RTX dependency.
./isaaclab.sh -i 'ov[ovphysx],rl[sb3]'
```

### 2. Install this task package into the same venv

```bash
# from wherever you rsync'd this directory to on Capella
pip install -e /path/to/aurova_bimanual_handover_isaaclab3
```

### 3. Register the task with Isaac Lab's training entrypoint

Isaac Lab discovers tasks by importing packages that call `gym.register(...)`
at import time (this package's `__init__.py` does that). The unified 3.0
training entrypoint doesn't auto-scan arbitrary external pip packages, so use
the small wrapper below to guarantee the import happens before training
starts — copy it next to Isaac Lab's own `scripts/reinforcement_learning/`:

```python
# ~/isaaclab3/scripts/reinforcement_learning/train_bimanual.py
import aurova_bimanual_handover  # noqa: F401 -- registers Isaac-Bimanual-Direct-reach-v0
import runpy
runpy.run_path(__file__.replace("train_bimanual.py", "train.py"), run_name="__main__")
```

### 4. SLURM job

Kit-less mode is a plain pip install — no Apptainer/Singularity container
and no `--nv` GPU driver mapping needed (that whole layer was the old
Vulkan-dependent workflow in `docker/cluster/`). A minimal job script:

```bash
#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:1
#SBATCH --time=04:00:00
#SBATCH --job-name=bimanual-handover

source ~/isaaclab3-venv/bin/activate
cd ~/isaaclab3

python scripts/reinforcement_learning/train_bimanual.py \
    --rl_library sb3 \
    --task Isaac-Bimanual-Direct-reach-v0 \
    --num_envs 1 \
    --headless
```

Submit with `sbatch job_bimanual.sh`. Start with `--num_envs 1` (matches this
task's current default and its `BimanualDirectCfg.num_envs = 1`) to first
confirm the port boots cleanly, then scale up.

### 5. Testing — do this interactively before ever submitting an sbatch job

Debugging a port through the SLURM queue (minutes of wait per iteration) is
slow. Get an interactive allocation first and iterate there; only move to
`sbatch` once stage 4 below passes.

```bash
# exact partition / gres name is cluster-specific — check `sinfo -o "%P %G"`
# or Capella's own docs; this is the generic SLURM pattern
srun --partition=<h100-partition> --gres=gpu:h100:1 --cpus-per-task=8 \
     --mem=32G --time=01:00:00 --pty bash

source ~/isaaclab3-venv/bin/activate
cd ~/isaaclab3
nvidia-smi   # sanity check the GPU is visible in the allocation
```

**Stage 1 — import only, no GPU/physics touched.** Cheapest possible check;
catches syntax errors, missing deps, and confirms `gym.register` ran.

```bash
python -c "import aurova_bimanual_handover; import gymnasium as gym; \
  print([k for k in gym.registry if 'Bimanual' in k])"
# expect: ['Isaac-Bimanual-Direct-reach-v0']
```

**Stage 2 — list/inspect the task through Isaac Lab's own tooling.** Check
what's actually in the clone first:

```bash
ls scripts/environments/
```

If `list_envs.py` is there, run it — it imports every registered task and
prints them, still without touching physics:

```bash
python scripts/environments/list_envs.py | grep Bimanual
```

**Stage 3 — step the env with no learning involved.** If `zero_agent.py` or
`random_agent.py` exists under `scripts/environments/`, use it — it goes
through Isaac Lab's real (Hydra-based) env construction path, so it's more
trustworthy than a hand-rolled script for catching API-signature mismatches:

```bash
python scripts/environments/zero_agent.py \
    --task Isaac-Bimanual-Direct-reach-v0 --num_envs 1 --headless
# or random_agent.py if zero_agent.py isn't present
```

Watch stdout/stderr for the specific things this port touched:
- Construction + first `_reset_idx` succeeds (exercises the split
  `write_joint_position_to_sim_index` / `write_joint_velocity_to_sim_index`
  calls and `write_root_pose_to_sim_index`).
- First `_pre_physics_step` / IK step runs without a shape mismatch
  (exercises `root_physx_view.get_jacobians()` — the one accessor left on
  its deprecated path; a jacobian shape error is the most likely failure
  here).
- No `NotImplementedError` or `AttributeError` from the asset/sensor API.

If neither script exists in this branch, fall back to a minimal manual
smoke test — same idea, just without Isaac Lab's Hydra config plumbing:

```bash
python scripts/reinforcement_learning/train_bimanual.py \
    --rl_library sb3 --task Isaac-Bimanual-Direct-reach-v0 \
    --num_envs 1 --headless --max_iterations 2
```

**Stage 4 — confirm it's actually running on `ovphysx`, not falling back
silently.** This is the whole point of the migration — verify it, don't
assume it:

```bash
pip show isaaclab_ovphysx
# and check the run's stdout/log for the physics backend it initialized —
# there should be no mention of Kit, RTX, or llvmpipe anywhere in the log
```

**Stage 5 — contact sensors under real contact.** The zero/random agent
won't reliably produce hand-object contact. Run a short real training smoke
test and check the logged metrics for nonzero force data (exercises
`ContactSensor.data.force_matrix_w.torch`):

```bash
python scripts/reinforcement_learning/train_bimanual.py \
    --rl_library sb3 --task Isaac-Bimanual-Direct-reach-v0 \
    --num_envs 4 --headless --max_iterations 50
# then check TensorBoard / stdout for metrics/contact_forces > 0 at some point
```

**Stage 6 — only now submit the real `sbatch` job**, scaling `--num_envs` up
from there.

If you'd rather isolate cluster/install issues from anything specific to
this port, bring up a plain PhysX task from Isaac Lab's own `isaaclab_tasks`
(e.g. `Isaac-Cartpole-Direct-v0`) through the same stages first — if that
fails at stage 1-4, the problem is the kit-less install, not this task.
