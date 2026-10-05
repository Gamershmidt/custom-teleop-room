#!/usr/bin/env bash
# One capture session on the messy_v3 scene set (narrow chaotic rooms with tables, mtc_capture/narrow_rooms.py):
#   mtc_capture/run_capture.sh OPERATOR HEIGHT_M [extra capture args, e.g. --space 5 3 --per-operator]
# Runs the Mac-side preflight (re-issuing the TLS certificate if the Mac's IP changed), then capture.
# Ctrl-C ends the session; the next run resumes where this one stopped.
set -euo pipefail
if [ $# -lt 2 ]; then
    echo "usage: $0 OPERATOR HEIGHT_M [capture args...]"; exit 2
fi
OPERATOR=$1; HEIGHT=$2; shift 2
REPO="$(cd "$(dirname "$0")/.." && pwd)"
# MTC_ENV: the environment script to source (default: the Mac teleop setup)
source "${MTC_ENV:-$HOME/Documents/teleop/env.sh}"
cd "$REPO"
SET=data/mtc_capture/scene_sets/messy_v3/manifest.json
# --space / --per-operator must match between preflight and capture
PRE=()
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
        --space) PRE+=(--space "${args[$i+1]}" "${args[$i+2]}") ;;
        --per-operator) PRE+=(--per-operator) ;;
        --reverse) PRE+=(--reverse) ;;
        --port) PRE+=(--port "${args[$i+1]}") ;;
        --native) PRE+=(--native) ;;
        --native-port) PRE+=(--native-port "${args[$i+1]}") ;;
    esac
done
FIX=(--fix-cert); [[ " $* " == *" --native "* ]] && FIX=()   # the native app needs no TLS certificate
python -m mtc_capture.preflight --operator "$OPERATOR" --operator-height "$HEIGHT" --scenes "$SET" ${FIX[@]+"${FIX[@]}"} ${PRE[@]+"${PRE[@]}"}
python -m mtc_capture.capture --operator "$OPERATOR" --operator-height "$HEIGHT" --scenes "$SET" "$@"
python -m mtc_capture.status --operator-height "$HEIGHT" 2>/dev/null | tail -1 || true
