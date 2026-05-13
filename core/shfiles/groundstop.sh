#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./core/shfiles/groundstop.sh [--uav uav0|uav1|uav2|all] [options]

Options:
  --uav TARGET            Target UAV, default: all
  --jobs N                Parallel stop jobs for groundctl stop, default: 3
  --land-timeout SECONDS  Max time to wait for land before continuing, default: 8
  --skip-land             Do not send land before stopping processes
  --no-force              Do not pass --force to groundctl stop
  -h, --help              Show this help

This script runs:
  groundctl.sh land --uav TARGET  # best effort, timeout protected
  groundctl.sh stop core --uav TARGET --force --jobs 3
  groundctl.sh stop vins-sync --uav TARGET --force --jobs 3
  groundctl.sh stop roscore --uav TARGET --force --jobs 3
  groundsync.sh stop
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUNDCTL="${SCRIPT_DIR}/groundctl.sh"
GROUNDSYNC="${SCRIPT_DIR}/groundsync.sh"
GROUNDGAME="${SCRIPT_DIR}/groundgame.sh"

UAV_TARGET="all"
JOBS="3"
LAND_TIMEOUT="8"
SKIP_LAND=0
FORCE_ARG="--force"
FAILED=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --uav)
      if [ "$#" -lt 2 ]; then
        printf '[groundstop] --uav requires a value\n' >&2
        exit 2
      fi
      UAV_TARGET="$2"
      shift 2
      ;;
    --uav=*)
      UAV_TARGET="${1#--uav=}"
      shift
      ;;
    --jobs)
      if [ "$#" -lt 2 ]; then
        printf '[groundstop] --jobs requires a value\n' >&2
        exit 2
      fi
      JOBS="$2"
      shift 2
      ;;
    --jobs=*)
      JOBS="${1#--jobs=}"
      shift
      ;;
    --land-timeout)
      if [ "$#" -lt 2 ]; then
        printf '[groundstop] --land-timeout requires a value\n' >&2
        exit 2
      fi
      LAND_TIMEOUT="$2"
      shift 2
      ;;
    --land-timeout=*)
      LAND_TIMEOUT="${1#--land-timeout=}"
      shift
      ;;
    --skip-land)
      SKIP_LAND=1
      shift
      ;;
    --no-force)
      FORCE_ARG=""
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf '[groundstop] unknown argument: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${UAV_TARGET}" in
  uav0|uav1|uav2|all)
    ;;
  *)
    printf '[groundstop] invalid --uav value: %s\n' "${UAV_TARGET}" >&2
    printf '[groundstop] expected one of: uav0, uav1, uav2, all\n' >&2
    exit 2
    ;;
esac

case "${JOBS}" in
  ''|*[!0-9]*)
    printf '[groundstop] --jobs must be a positive integer, got: %s\n' "${JOBS}" >&2
    exit 2
    ;;
  0)
    printf '[groundstop] --jobs must be greater than 0\n' >&2
    exit 2
    ;;
esac

if [[ ! "${LAND_TIMEOUT}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  printf '[groundstop] --land-timeout must be a positive number, got: %s\n' "${LAND_TIMEOUT}" >&2
  exit 2
fi

if [ ! -x "${GROUNDCTL}" ]; then
  printf '[groundstop] missing executable: %s\n' "${GROUNDCTL}" >&2
  exit 1
fi

if [ ! -x "${GROUNDSYNC}" ]; then
  printf '[groundstop] missing executable: %s\n' "${GROUNDSYNC}" >&2
  exit 1
fi

print_command() {
  printf '[groundstop] command:'
  printf ' %q' "$@"
  printf '\n'
}

run_best_effort_land() {
  if [ "${SKIP_LAND}" -eq 1 ]; then
    printf '\n[groundstop] skip land\n'
    return 0
  fi

  printf '\n[groundstop] best-effort land on %s, timeout=%ss\n' "${UAV_TARGET}" "${LAND_TIMEOUT}"
  if command -v timeout >/dev/null 2>&1; then
    print_command timeout --kill-after=2s "${LAND_TIMEOUT}s" "${GROUNDCTL}" land --uav "${UAV_TARGET}" --connect-timeout 1
    timeout --kill-after=2s "${LAND_TIMEOUT}s" "${GROUNDCTL}" land --uav "${UAV_TARGET}" --connect-timeout 1
  else
    printf '[groundstop] warning: timeout command not found; skip land to avoid blocking\n' >&2
    return 0
  fi

  local rc=$?
  if [ "${rc}" -eq 0 ]; then
    printf '[groundstop] land command finished\n'
  elif [ "${rc}" -eq 124 ] || [ "${rc}" -eq 137 ]; then
    printf '[groundstop] land command timed out; continue stopping\n' >&2
  else
    printf '[groundstop] land command failed with exit code %s; continue stopping\n' "${rc}" >&2
  fi
}

run_game_stop() {
  if [ ! -x "${GROUNDGAME}" ]; then
    return 0
  fi
  printf '\n[groundstop] stop game controllers\n'
  print_command "${GROUNDGAME}" stop --uav "${UAV_TARGET}" --force --jobs "${JOBS}"
  "${GROUNDGAME}" stop --uav "${UAV_TARGET}" --force --jobs "${JOBS}"
  local rc=$?
  if [ "${rc}" -ne 0 ]; then
    printf '[groundstop] stop game controllers failed with exit code %s; continue stopping\n' "${rc}" >&2
    FAILED=1
  fi
}

run_stop_step() {
  local target="$1"
  shift
  local args=()
  printf '\n[groundstop] stop %s on %s\n' "${target}" "${UAV_TARGET}"
  if [ -n "${FORCE_ARG}" ]; then
    args+=("${FORCE_ARG}")
  fi
  print_command "${GROUNDCTL}" stop "${target}" --uav "${UAV_TARGET}" "${args[@]}" --jobs "${JOBS}"
  "${GROUNDCTL}" stop "${target}" --uav "${UAV_TARGET}" "${args[@]}" --jobs "${JOBS}"
  local rc=$?
  if [ "${rc}" -ne 0 ]; then
    printf '[groundstop] stop %s failed with exit code %s; continue stopping\n' "${target}" "${rc}" >&2
    FAILED=1
  fi
}

run_groundsync_stop() {
  printf '\n[groundstop] stop ground sync\n'
  print_command "${GROUNDSYNC}" stop
  "${GROUNDSYNC}" stop
  local rc=$?
  if [ "${rc}" -ne 0 ]; then
    printf '[groundstop] groundsync stop failed with exit code %s\n' "${rc}" >&2
    FAILED=1
  fi
}

run_game_stop
run_best_effort_land
run_stop_step core
run_stop_step vins-sync
run_stop_step roscore
run_groundsync_stop

if [ "${FAILED}" -ne 0 ]; then
  printf '\n[groundstop] finished with stop errors\n' >&2
  exit 1
fi

printf '\n[groundstop] finished\n'
