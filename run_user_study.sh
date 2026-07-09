#!/usr/bin/env bash
set -euo pipefail

is_frontend_stack() {
  case "${1:-}" in
    classic|lego|sandwich|hybrid) return 0 ;;
    *) return 1 ;;
  esac
}

is_block_id() {
  case "${1:-}" in
    B[0-9]*|b[0-9]*|main|practice) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ $# -lt 2 || $# -gt 5 ]]; then
  cat <<'EOF'
Usage:
  ./run_user_study.sh <participant_id> <condition_id> [session_id] [frontend_stack] [block_id]
  ./run_user_study.sh <participant_id> <condition_id> <block_id> [session_id] [frontend_stack]

Examples:
  ./run_user_study.sh P01 optimized
  ./run_user_study.sh P01 optimized pilot_20260707 sandwich
  ./run_user_study.sh P02 unoptimized main pilot_20260707 sandwich

Behavior:
  - sources ROS Noetic and ~/catkin_ws/devel/setup.bash
  - defaults dashboard host to 0.0.0.0
  - defaults dashboard port to 8766
  - auto-generates session_id when omitted
  - defaults frontend_stack to classic
  - keeps block_id optional for compatibility with older study logs
EOF
  exit 2
fi

PARTICIPANT_ID="$1"
CONDITION_ID="$2"
SESSION_ID="pilot_$(date +%Y%m%d)"
FRONTEND_STACK="classic"
BLOCK_ID=""

if [[ $# -ge 3 ]]; then
  if is_block_id "$3"; then
    BLOCK_ID="$3"
    if [[ $# -ge 4 ]]; then
      SESSION_ID="$4"
    fi
    if [[ $# -ge 5 ]]; then
      FRONTEND_STACK="$5"
    fi
  else
    SESSION_ID="$3"
    if [[ $# -ge 4 ]]; then
      if is_frontend_stack "$4"; then
        FRONTEND_STACK="$4"
      else
        BLOCK_ID="$4"
      fi
    fi
    if [[ $# -ge 5 ]]; then
      if is_frontend_stack "$5"; then
        FRONTEND_STACK="$5"
      else
        BLOCK_ID="$5"
      fi
    fi
  fi
fi

cd "$HOME/catkin_ws"
export PATH=/usr/bin:/bin:/usr/sbin:/sbin:/opt/ros/noetic/bin:$PATH
source /opt/ros/noetic/setup.bash
source "$HOME/catkin_ws/devel/setup.bash"

echo "Launching user study with:"
echo "  session_id=$SESSION_ID"
echo "  participant_id=$PARTICIPANT_ID"
echo "  condition_id=$CONDITION_ID"
echo "  block_id=${BLOCK_ID:-<optional>}"
echo "  frontend_stack=$FRONTEND_STACK"
if [[ "$FRONTEND_STACK" == "sandwich" ]]; then
  echo "  use_manual_sandwich_labeler=true"
fi
echo "  dashboard_host=0.0.0.0"
echo "  dashboard_port=8766"
echo

EXTRA_ARGS=()
if [[ "$FRONTEND_STACK" == "sandwich" ]]; then
  EXTRA_ARGS+=(
    use_manual_sandwich_labeler:=true
    launch_sam_sandwich:=true
  )
fi

exec roslaunch tabletop_workspace_opt user_study.launch \
  user_study_dashboard_host:=0.0.0.0 \
  user_study_dashboard_port:=8766 \
  session_id:="$SESSION_ID" \
  participant_id:="$PARTICIPANT_ID" \
  condition_id:="$CONDITION_ID" \
  block_id:="$BLOCK_ID" \
  frontend_stack:="$FRONTEND_STACK" \
  "${EXTRA_ARGS[@]}"
