#!/bin/bash
# Run shared autonomy for all permutations of pick objects.
#
# Prerequisites: sim must be running in another terminal:
#   roslaunch tabletop_workspace_opt sim_moveit.launch scene_name:=scene_desk
#
# Usage:
#   # Baseline layout:
#   bash scripts/eval/run_sa_all_orderings.sh config/tasks/desk_organize_v2_sa.yaml mug,stapler,pen_cup
#
#   # Optimized layout (launch sim with scene_desk_me_optimized first):
#   bash scripts/eval/run_sa_all_orderings.sh config/tasks/desk_organize_v2_sa.yaml mug,stapler,pen_cup scene_desk_me_optimized

set -e

TASK_CONFIG="${1:?Usage: $0 <task_config.yaml> <obj1,obj2,obj3> [scene_name]}"
OBJECTS="${2:?Provide comma-separated pick objects, e.g. mug,stapler,pen_cup}"
SCENE="${3:-}"

NOISE="${NOISE:-0.03}"
THRESHOLD="${THRESHOLD:-0.8}"
BETA="${BETA:-5.0}"
MAX_SPEED="${MAX_SPEED:-0.05}"

IFS=',' read -ra OBJ_ARRAY <<< "$OBJECTS"
N=${#OBJ_ARRAY[@]}
NUM_PERMS=$(python3 -c "import math; print(math.factorial($N))")

echo "============================================================"
echo "  SA Benchmark: All orderings of [${OBJECTS}]"
echo "  Task: ${TASK_CONFIG}"
echo "  Scene: ${SCENE:-from task config}"
echo "  Objects: ${N} -> ${NUM_PERMS} orderings"
echo "============================================================"
echo ""

# Generate all permutations
PERMS=$(python3 -c "
from itertools import permutations
objs = '${OBJECTS}'.split(',')
for p in permutations(objs):
    print(','.join(p))
")

COUNT=0

for PERM in $PERMS; do
    COUNT=$((COUNT + 1))
    echo ""
    echo "=== Ordering $COUNT/$NUM_PERMS: $PERM ==="

    SCENE_ARG=""
    if [ -n "$SCENE" ]; then
        SCENE_ARG="scene:=$SCENE"
    fi

    roslaunch tabletop_workspace_opt shared_autonomy.launch \
        task_config:="$TASK_CONFIG" \
        noise:="$NOISE" \
        threshold:="$THRESHOLD" \
        max_speed:="$MAX_SPEED" \
        user_sequence:="$PERM" \
        $SCENE_ARG \
        2>&1

    echo "=== Done ordering $COUNT/$NUM_PERMS ==="
done

echo ""
echo "============================================================"
echo "  All $NUM_PERMS orderings complete."
echo "  Results saved in results/sa_runs/"
echo "============================================================"
