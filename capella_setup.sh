#!/bin/bash
# Idempotent Capella setup for the bimanual-handover Isaac Lab 3.0 kit-less install.
# Run on a compute node (NOT the login node -- glibc there is too old, see below),
# outside the Apptainer container: this script builds the container image and venv
# fresh, then does the container-side install itself.
#
# Usage: bash capella_setup.sh
set -euo pipefail

WS=/data/horse/ws/hapi039h-handover
ISAACLAB="$WS/isaaclab3"
VENV="$WS/isaaclab3-venv"
SIF="$WS/ubuntu2204-cuda.sif"
TASK_REPO="$WS/aurova_bimanual_handover_isaaclab3"
TASK_GIT_URL="git@github.com:<your-username>/aurova_bimanual_handover_isaaclab3.git"

echo "[1/6] Apptainer container image"
if [ ! -f "$SIF" ]; then
    apptainer pull "$SIF" docker://nvcr.io/nvidia/cuda:12.8.1-base-ubuntu22.04
else
    echo "  already present, skipping"
fi

echo "[2/6] Isaac Lab 3.0 clone"
if [ ! -d "$ISAACLAB" ]; then
    git clone -b release/3.0.0-beta2 https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB"
else
    echo "  already present, skipping clone (not pulling -- avoid clobbering local patches)"
fi

echo "[3/6] Patch glibc-2.34-incompatible setup.py pins"
cd "$ISAACLAB"
sed -i '/omniverseclient==2.71.1.7015/d' source/isaaclab/setup.py
sed -i '/usd-exchange>=2.2/d' source/isaaclab/setup.py

echo "[4/6] venv"
if [ ! -d "$VENV" ]; then
    python3 -m venv "$VENV"
else
    echo "  already present, skipping"
fi

echo "[5/6] Task package clone"
if [ ! -d "$TASK_REPO" ]; then
    git clone "$TASK_GIT_URL" "$TASK_REPO"
else
    echo "  already present -- pulling latest"
    (cd "$TASK_REPO" && git pull)
fi

echo "[6/6] Container-side install (this is the slow part, several minutes)"
apptainer exec --nv "$SIF" bash -c "
    set -euo pipefail
    source '$VENV/bin/activate'
    cd '$ISAACLAB'
    ./isaaclab.sh -i 'ov[ovphysx],rl[sb3]'
    pip install -e '$TASK_REPO'

    mkdir -p scripts/reinforcement_learning scripts/environments
    cat > scripts/reinforcement_learning/train_bimanual.py << 'EOF'
import aurova_bimanual_handover  # noqa: F401
import runpy
runpy.run_path(__file__.replace('train_bimanual.py', 'train.py'), run_name='__main__')
EOF
    cat > scripts/environments/zero_agent_bimanual.py << 'EOF'
import aurova_bimanual_handover  # noqa: F401
import runpy
runpy.run_path(__file__.replace('zero_agent_bimanual.py', 'zero_agent.py'), run_name='__main__')
EOF
"

echo "Done. To work interactively:"
echo "  apptainer exec --nv $SIF bash"
echo "  source $VENV/bin/activate"
echo "  cd $ISAACLAB"
