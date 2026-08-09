# Bimanual Handover — Isaac Lab 3.0 port

Ported from the `aurova_reinforcement_learning/bimanual_handover` task in the old
`omni.isaac.lab_tasks` (Isaac Lab 1.2.0) repo. Target: Isaac Lab 3.0 (beta2),
**OvPhysX backend** (kit-less), so it runs on Capella's H100 nodes without
needing a full Isaac Sim / Omniverse Kit install — see "why" below.

This directory is a standalone, pip-installable task package — it does **not**
contain Isaac Lab itself. Isaac Lab 3.0 gets cloned separately on the cluster.

**Status: blocked on a robot USD asset compatibility issue** (as of last
session). Config/env code itself is confirmed working — `zero_agent.py` got
past env construction, past the ground-plane/marker fixes, and failed only
when spawning the actual robots. See "Blocked: legacy USD asset
compatibility" below for the full diagnosis and where to pick back up.

## Why kit-less, not full Isaac Sim

Isaac Sim's Kit app boots a Vulkan/EGL-based renderer at startup even for
headless, no-camera RL training. Capella's H100 nodes are provisioned as
compute-only (no GL/Vulkan driver stack for a display-less datacenter GPU),
so Kit was silently falling back to Mesa's `llvmpipe` software rasterizer —
the real cause of the original `LLVM: out of memory` crash, independent of
`--mem`/GPU exclusivity.

Isaac Lab 3.0 splits physics from rendering into separate installable
packages, one of which (`ov[ovphysx]`) is a standalone PhysX runtime wheel
needing no Kit, no Vulkan, no display driver — pure CUDA.

**Important distinction discovered while debugging this on Capella**: there
are *two* separate PhysX config classes, and they are not interchangeable:

- `isaaclab_physx.physics.PhysxCfg` — the full **Isaac-Sim/Kit-based** PhysX.
  Setting `SimulationCfg(physics=PhysxCfg())` forces Isaac Lab's launcher to
  require Kit (`needs_kit=True` in `isaaclab_tasks.utils.sim_launcher`), which
  then hard-fails with no Isaac Sim installed. This is what the port
  originally (incorrectly) used.
- `isaaclab_ovphysx.physics.OvPhysxCfg` — the actual **kit-less** runtime,
  matching what `ov[ovphysx]` installs. This is what `bimanual_direct_env_cfg.py`
  uses now.

Not used here: the Newton/Warp backend. Isaac Lab's Newton docs currently only
confirm classic-RL / flat-terrain-locomotion coverage — this task is
contact-rich dual-arm + underactuated-hand manipulation (17 contact sensors,
custom force-based rewards), so it stays on PhysX (OvPhysX).

## What changed in the port

- `omni.isaac.lab*` → `isaaclab*` / `isaaclab_ovphysx` imports (Isaac Lab 3.0
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
  `write_root_pose_to_sim` → `write_root_pose_to_sim_index`.
- `SimulationCfg` takes an explicit `physics=OvPhysxCfg()` (from
  `isaaclab_ovphysx.physics`) — see the distinction called out above.
- `from isaaclab.utils import configclass` → `from isaaclab.utils.configclass
  import configclass`. `isaaclab/utils/__init__.py` no longer re-exports
  `configclass` at the package level in 3.0, so the old import silently binds
  the *submodule* instead of the decorator function
  (`TypeError: 'module' object is not callable` on first `@configclass` use).
- Fixed a hardcoded absolute USD path
  (`/home/hapi039h/isaaclab/...`, a leftover from a previous cluster user)
  in `robots_cfg.py` — now resolved relative to the package, via
  `assets/config/usd/`.
- Fixed a pre-existing bug in `mdp/__init__.py`: `from utils import *` (missing
  the relative dot — would have failed at import) → `from .utils import *`.

## What was *not* changed (left as deprecated-but-working, or unverified)

- `root_physx_view.get_jacobians()` in `_get_ee_pose()` — the property is
  deprecated in favor of `root_view` / `data.body_link_jacobian_w`, but still
  works today.
- `ContactSensor.data.force_matrix_w` — assumed unchanged; not yet exercised
  by a real contact event on Capella (zero-agent testing doesn't produce
  contact). **Validate this next** — see Testing, stage 5.
- Camera path (`self.scene["camera"]`, `save_images_grid`) — `render_imgs`
  defaults to `False`, so this code path is inactive by default. There's also
  no `camera` sensor actually registered in `_setup_scene()` in the original
  code — a pre-existing latent bug, not something the migration introduced.

## Isaac Lab 3.0 beta2 rough edges hit on Capella (not this task's bugs)

These are packaging/beta issues in `release/3.0.0-beta2` itself, worth
knowing about since they'll resurface on any similarly-configured cluster.
Fixes are folded into the setup steps below.

1. **`omniverseclient==2.71.1.7015`** is an unconditional dependency of core
   `isaaclab` (for the Kit livestream feature only), but has no wheel below
   glibc 2.35. Fix: delete the line from `source/isaaclab/setup.py`.
2. **`usd-exchange>=2.2`** — same situation, and not imported anywhere in the
   `isaaclab` core package. Fix: delete the line from `source/isaaclab/setup.py`.
3. **`ovphysx==0.4.13`** itself has no wheel below glibc 2.35. Unlike the two
   above, this is the actual payload — can't be deleted. Capella's OS glibc is
   2.34 on both login and compute nodes, so the whole install (and later,
   training) has to run inside an Apptainer container with a newer glibc
   userland (Ubuntu 22.04+). See setup step 1.
4. **`--headless` doesn't suppress the default Kit visualizer in time** for an
   internal compatibility check (`validate_runtime_compatibility` in
   `isaaclab_tasks/utils/sim_launcher.py`) — it raises "OvPhysX physics is
   kitless and cannot be used together with the Kit visualizer" even when no
   `--visualizer kit` was passed, because the CLI's `--visualizer` default
   isn't cleared by `--headless` before that check runs. Fix: pass
   `--viz none` (or `--visualizer none`) explicitly on every invocation —
   don't rely on `--headless` alone.
5. External task packages aren't auto-imported by Isaac Lab's own scripts
   (`train.py`, `zero_agent.py`, etc.) — see step 4 below for the wrapper
   pattern needed to work around this.

## Blocked: legacy USD asset compatibility (current status)

`zero_agent_bimanual.py` gets past env construction, the ground-plane spawn,
and the debug markers (both fixed — see above), then fails spawning the
actual robots:

```
Encountered: std::bad_alloc, while reading @.../ur5e_4f_ros2.usd@
```

**Diagnosis, fully isolated:**

- Not corruption: sha256 of the file on Capella matches the original 1.2.0
  repo exactly.
- Not `usd-core` 25.11-specific: `usd-core==24.11` (tested locally on macOS)
  and `usd-exchange`'s bundled Omniverse-CI build (`pxrInternal_v0_25_5`,
  tested on Capella) both fail identically on the same files.
- Not an environment/memory-limit issue: a trivial file created *and read
  back* with the exact same `usd-core` 25.11, in the exact same container/venv
  that fails on the robot files, works fine — for both `.usda` (ASCII) and
  `.usdc` (binary crate) formats.
- Both robot USDs fail (`gen3_4f.usd`, 6MB; `ur5e_4f_ros2.usd`, 27MB) — not
  file-size-specific.
- The files are "USD crate, version 0.8.0" — a genuinely old format (roughly
  2019-2020-era Pixar USD). This task's ~30M-step SB3 checkpoints prove they
  loaded fine under whatever Isaac Sim/Kit version originally trained it.

**Conclusion**: every 2025-era USD build tried (3 so far, from 2 different
vendors) fails on these specific old files; only the original, much older
Isaac Sim/Kit runtime is known to read them correctly.

**In progress, not yet working**: pulling an old Isaac Sim Docker image via
Apptainer (`nvcr.io/nvidia/isaac-sim:4.2.0`, guessed to roughly match this
repo's pinned Isaac Lab 1.2.0 era) to use *only its bundled `pxr` library*
directly — no `SimulationApp`/Kit app launch, so no Vulkan/renderer
requirement — to re-save the two files into a format the modern kit-less
reader can open. Progress so far:

```bash
apptainer pull /data/horse/ws/hapi039h-handover/isaac-sim-4.2.sif docker://nvcr.io/nvidia/isaac-sim:4.2.0
apptainer exec --nv --bind /data:/data /data/horse/ws/hapi039h-handover/isaac-sim-4.2.sif bash
source /isaac-sim/setup_conda_env.sh   # confirmed: this correctly populates PYTHONPATH
# but: `python`, `python.sh`, and `/isaac-sim/python.sh` all report "no python" --
# not yet resolved. Possible causes to check next: this NGC tag may not bundle a
# full Python runtime the way assumed; wrong container tag entirely; or the
# python binary needs a different invocation path than guessed. Start by just
# `ls /isaac-sim/` and looking for whatever the actual Python entry point is
# (find . -iname "python*" -maxdepth 2 from /isaac-sim), rather than assuming
# python.sh is it.
```

**Fallback if this doesn't pan out**: ask whoever originally trained this task
(their working Isaac Sim setup can open these files fine) to re-save
`ur5e_4f_ros2.usd` and `gen3_4f.usd` and send them back — a ~10 minute favor,
see the conversation history for exactly what to ask for. Regenerating from
the URDF sources in `assets/config/urdf/` via `usd-exchange`'s authoring API
is the other fallback, but it's substantial new engineering, not a quick fix.

## Setup on Capella

### 1. Apptainer container (needed because of rough edge #3 above)

```bash
cd /data/horse/ws/hapi039h-handover
apptainer pull ubuntu2204-cuda.sif docker://nvcr.io/nvidia/cuda:12.8.1-base-ubuntu22.04
```

Everything from here on — install *and* later training/testing — runs inside
this container via `apptainer exec --nv --bind /data:/data ubuntu2204-cuda.sif bash`, since the
venv ends up with glibc-2.35-linked binaries that won't run on the bare host
(2.34). `git` isn't in this minimal image; clone repos from the host, only
`pip`/`python` need to run inside the container.

### 2. Clone Isaac Lab 3.0, patch two setup.py pins, install kit-less

```bash
# on the host (git isn't in the container image)
git clone -b release/3.0.0-beta2 https://github.com/isaac-sim/IsaacLab.git /data/horse/ws/hapi039h-handover/isaaclab3
cd /data/horse/ws/hapi039h-handover/isaaclab3
sed -i '/omniverseclient==2.71.1.7015/d' source/isaaclab/setup.py
sed -i '/usd-exchange>=2.2/d' source/isaaclab/setup.py

python3 -m venv /data/horse/ws/hapi039h-handover/isaaclab3-venv

# now enter the container for everything else
apptainer exec --nv --bind /data:/data /data/horse/ws/hapi039h-handover/ubuntu2204-cuda.sif bash
source /data/horse/ws/hapi039h-handover/isaaclab3-venv/bin/activate
cd /data/horse/ws/hapi039h-handover/isaaclab3
./isaaclab.sh -i 'ov[ovphysx],rl[sb3]'
```

### 3. Install this task package into the same venv (still inside the container)

```bash
# clone on the host first (no git in the container), then pip install inside it
# (run on host:)
git clone git@github.com:<your-username>/aurova_bimanual_handover_isaaclab3.git \
    /data/horse/ws/hapi039h-handover/aurova_bimanual_handover_isaaclab3
# (back inside the container:)
pip install -e /data/horse/ws/hapi039h-handover/aurova_bimanual_handover_isaaclab3
```

### 4. Register the task with Isaac Lab's training entrypoint

Isaac Lab discovers tasks by importing packages that call `gym.register(...)`
at import time (this package's `__init__.py` does that). Isaac Lab's own
scripts (`train.py`, `zero_agent.py`, `random_agent.py`, ...) don't auto-scan
arbitrary external pip packages, so wrap each one you need with a one-line
import first. Pattern (swap the filename for whichever script you're using):

```bash
cat > scripts/reinforcement_learning/train_bimanual.py << 'EOF'
import aurova_bimanual_handover  # noqa: F401 -- registers Isaac-Bimanual-Direct-reach-v0
import runpy
runpy.run_path(__file__.replace("train_bimanual.py", "train.py"), run_name="__main__")
EOF

cat > scripts/environments/zero_agent_bimanual.py << 'EOF'
import aurova_bimanual_handover  # noqa: F401
import runpy
runpy.run_path(__file__.replace("zero_agent_bimanual.py", "zero_agent.py"), run_name="__main__")
EOF
```

### 5. SLURM job

```bash
#!/bin/bash
#SBATCH --account=p_lv_ra_2526
#SBATCH --partition=capella
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=bimanual-handover

WS=/data/horse/ws/hapi039h-handover

apptainer exec --nv --bind /data:/data "$WS/ubuntu2204-cuda.sif" bash -c "
    source $WS/isaaclab3-venv/bin/activate
    cd $WS/isaaclab3
    python scripts/reinforcement_learning/train_bimanual.py \
        --rl_library sb3 \
        --task Isaac-Bimanual-Direct-reach-v0 \
        --num_envs 1 \
        --headless --viz none
"
```

Submit with `sbatch job_bimanual.sh`. Start with `--num_envs 1` (matches this
task's current default) to first confirm the port boots cleanly, then scale
up. Note the `--viz none` — required, see rough edge #4 above.

## Testing — do this interactively before ever submitting an sbatch job

Debugging a port through the SLURM queue (minutes of wait per iteration) is
slow. Get an interactive allocation and iterate there; only move to `sbatch`
once stage 5 below passes.

```bash
srun --account=p_lv_ra_2526 --partition=capella --nodes=1 --gpus=1 \
     --cpus-per-task=8 --mem=32G --time=01:00:00 --pty bash

apptainer exec --nv --bind /data:/data /data/horse/ws/hapi039h-handover/ubuntu2204-cuda.sif bash
source /data/horse/ws/hapi039h-handover/isaaclab3-venv/bin/activate
cd /data/horse/ws/hapi039h-handover/isaaclab3
nvidia-smi   # sanity check the GPU is visible in the allocation
```

**Stage 1 — import only, no GPU/physics touched.**

```bash
python -c "import aurova_bimanual_handover; import gymnasium as gym; \
  print([k for k in gym.registry if 'Bimanual' in k])"
# expect: ['Isaac-Bimanual-Direct-reach-v0']
```

**Stage 2 — list the task through Isaac Lab's own tooling** (`list_envs.py`
doesn't need the external-package wrapper since it just enumerates whatever's
already registered by whoever imports it — use the same wrapper pattern if it
comes up empty).

```bash
ls scripts/environments/
python scripts/environments/list_envs.py | grep Bimanual
```

**Stage 3 — step the env with no learning involved.** ✅ Confirmed working:

```bash
python scripts/environments/zero_agent_bimanual.py \
    --task Isaac-Bimanual-Direct-reach-v0 --num_envs 1 --headless --viz none
```

This exercises the split `write_joint_position_to_sim_index` /
`write_joint_velocity_to_sim_index` calls, `write_root_pose_to_sim_index`, and
the IK step's `root_physx_view.get_jacobians()` call — all passed.

**Stage 4 — confirm it's actually running on `ovphysx`, not falling back
silently.**

```bash
pip show isaaclab_ovphysx
# check stage 3's output for any mention of Kit, RTX, or llvmpipe — there
# shouldn't be any
```

**Stage 5 — contact sensors under real contact (not yet run).** The
zero-agent won't reliably produce hand-object contact. Run a short real
training smoke test and check the logged metrics for nonzero force data
(exercises `ContactSensor.data.force_matrix_w.torch`):

```bash
python scripts/reinforcement_learning/train_bimanual.py \
    --rl_library sb3 --task Isaac-Bimanual-Direct-reach-v0 \
    --num_envs 4 --headless --viz none --max_iterations 50
# then check TensorBoard / stdout for metrics/contact_forces > 0 at some point
```

**Stage 6 — only now submit the real `sbatch` job**, scaling `--num_envs` up
from there.
