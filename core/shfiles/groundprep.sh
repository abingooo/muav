#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./core/shfiles/groundprep.sh [--uav uav0|uav1|uav2|all] [options]

Options:
  --uav TARGET          Target UAV, default: all
  --start-rviz BOOL     Start RViz through groundsync, default: false
  --no-ego              Pass --no-ego to "groundctl start core"
  --no-wait-ready       Pass --no-wait-ready to all groundctl start steps
  --strict-ready        Pass --strict-ready to all groundctl start steps
  -h, --help            Show this help

This script runs:
  groundctl.sh start roscore --uav TARGET
  groundctl.sh start core --uav TARGET
  groundctl.sh start vins-sync --uav TARGET
  groundsync.sh sync start_rviz:=false
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUNDCTL="${SCRIPT_DIR}/groundctl.sh"
GROUNDSYNC="${SCRIPT_DIR}/groundsync.sh"

UAV_TARGET="all"
START_RVIZ="false"
CORE_ARGS=()
START_ARGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --uav)
      if [ "$#" -lt 2 ]; then
        printf '[groundprep] --uav requires a value\n' >&2
        exit 2
      fi
      UAV_TARGET="$2"
      shift 2
      ;;
    --uav=*)
      UAV_TARGET="${1#--uav=}"
      shift
      ;;
    --start-rviz)
      if [ "$#" -lt 2 ]; then
        printf '[groundprep] --start-rviz requires true or false\n' >&2
        exit 2
      fi
      START_RVIZ="$2"
      shift 2
      ;;
    --start-rviz=*)
      START_RVIZ="${1#--start-rviz=}"
      shift
      ;;
    --no-ego)
      CORE_ARGS+=("--no-ego")
      shift
      ;;
    --no-wait-ready)
      START_ARGS+=("--no-wait-ready")
      shift
      ;;
    --strict-ready)
      START_ARGS+=("--strict-ready")
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf '[groundprep] unknown argument: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${UAV_TARGET}" in
  uav0|uav1|uav2|all)
    ;;
  *)
    printf '[groundprep] invalid --uav value: %s\n' "${UAV_TARGET}" >&2
    printf '[groundprep] expected one of: uav0, uav1, uav2, all\n' >&2
    exit 2
    ;;
esac

case "${START_RVIZ}" in
  true|false)
    ;;
  True|TRUE|1|yes|Yes|YES)
    START_RVIZ="true"
    ;;
  False|FALSE|0|no|No|NO)
    START_RVIZ="false"
    ;;
  *)
    printf '[groundprep] invalid --start-rviz value: %s\n' "${START_RVIZ}" >&2
    printf '[groundprep] expected true or false\n' >&2
    exit 2
    ;;
esac

if [ ! -x "${GROUNDCTL}" ]; then
  printf '[groundprep] missing executable: %s\n' "${GROUNDCTL}" >&2
  exit 1
fi

if [ ! -x "${GROUNDSYNC}" ]; then
  printf '[groundprep] missing executable: %s\n' "${GROUNDSYNC}" >&2
  exit 1
fi

run_step() {
  local label="$1"
  shift
  printf '\n[groundprep] %s\n' "${label}"
  printf '[groundprep] command:'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

run_step "start roscore on ${UAV_TARGET}" \
  "${GROUNDCTL}" start roscore --uav "${UAV_TARGET}" "${START_ARGS[@]}"

run_step "start core on ${UAV_TARGET}" \
  "${GROUNDCTL}" start core --uav "${UAV_TARGET}" "${CORE_ARGS[@]}" "${START_ARGS[@]}"

run_step "start vins-sync on ${UAV_TARGET}" \
  "${GROUNDCTL}" start vins-sync --uav "${UAV_TARGET}" "${START_ARGS[@]}"

printf '\n[groundprep] start ground sync, start_rviz=%s\n' "${START_RVIZ}"
printf '[groundprep] command: %q %q %q\n' "${GROUNDSYNC}" sync "start_rviz:=${START_RVIZ}"
exec "${GROUNDSYNC}" sync "start_rviz:=${START_RVIZ}"
