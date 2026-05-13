#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./core/shfiles/groundgame.sh start [--defender0 uav0] [--defender1 uav1] [--enemy uav2]
  ./core/shfiles/groundgame.sh single --uav uav1 --role defender_0
  ./core/shfiles/groundgame.sh stop [--uav all] [--force] [--jobs 3]
  ./core/shfiles/groundgame.sh status [--uav all]

Options:
  --defender0 UAV          UAV assigned to defender_0, default: uav0
  --defender1 UAV          UAV assigned to defender_1, default: uav1
  --enemy UAV              UAV assigned to enemy and MPC, default: uav2
  --role ROLE              Single-test role: defender_0, defender_1, or enemy
  --game-end-enabled VAL   Passed to adv/mpc launch, default: auto
  --uav TARGET             stop/status target: uav0,uav1,uav2, comma list, or all
  --jobs N                 Parallel SSH jobs for stop/status, default: 3
  --force                  Stop uses SIGKILL after SIGTERM
  -h, --help               Show this help

Start behavior:
  defender_0 UAV runs ADV
  defender_1 UAV runs ADV
  enemy UAV runs MPC
  fleet_uavs is built as defender0,defender1,enemy

Single behavior:
  Only the selected --uav starts one process.
  defender_0/defender_1 runs ADV with explicit output_role.
  enemy runs MPC.
USAGE
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="$(cd "${CORE_DIR}/.." && pwd)"
CONFIG_FILE="${PROJECT_DIR}/core/src/groundctrl/config/groundctrl.yaml"

COMMAND="${1:-}"
if [[ "${COMMAND}" == "-h" || "${COMMAND}" == "--help" ]]; then
  usage
  exit 0
fi
if [ "$#" -gt 0 ]; then
  shift
fi

DEFENDER0="uav0"
DEFENDER1="uav1"
ENEMY="uav2"
GAME_ROLE="defender_0"
GAME_END_ENABLED="auto"
TARGET_SPEC="all"
JOBS="3"
FORCE=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --defender0)
      DEFENDER0="${2:-}"
      shift 2
      ;;
    --defender0=*)
      DEFENDER0="${1#--defender0=}"
      shift
      ;;
    --defender1)
      DEFENDER1="${2:-}"
      shift 2
      ;;
    --defender1=*)
      DEFENDER1="${1#--defender1=}"
      shift
      ;;
    --enemy)
      ENEMY="${2:-}"
      shift 2
      ;;
    --enemy=*)
      ENEMY="${1#--enemy=}"
      shift
      ;;
    --role)
      GAME_ROLE="${2:-}"
      shift 2
      ;;
    --role=*)
      GAME_ROLE="${1#--role=}"
      shift
      ;;
    --game-end-enabled)
      GAME_END_ENABLED="${2:-}"
      shift 2
      ;;
    --game-end-enabled=*)
      GAME_END_ENABLED="${1#--game-end-enabled=}"
      shift
      ;;
    --uav)
      TARGET_SPEC="${2:-}"
      shift 2
      ;;
    --uav=*)
      TARGET_SPEC="${1#--uav=}"
      shift
      ;;
    --jobs)
      JOBS="${2:-}"
      shift 2
      ;;
    --jobs=*)
      JOBS="${1#--jobs=}"
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf '[groundgame] unknown argument: %s\n\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${COMMAND}" != "start" && "${COMMAND}" != "single" && "${COMMAND}" != "stop" && "${COMMAND}" != "status" ]]; then
  usage >&2
  exit 2
fi

case "${JOBS}" in
  ''|*[!0-9]*|0)
    printf '[groundgame] --jobs must be a positive integer, got: %s\n' "${JOBS}" >&2
    exit 2
    ;;
esac

CONFIG_OUTPUT="$(
  python3 - "${CONFIG_FILE}" "${TARGET_SPEC}" "${DEFENDER0}" "${DEFENDER1}" "${ENEMY}" "${COMMAND}" "${GAME_ROLE}" <<'PY'
import sys
import yaml

config_path, target_spec, defender0, defender1, enemy, command, game_role = sys.argv[1:8]
with open(config_path, "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

ordered = sorted(config["uavs"], key=lambda name: int(config["uavs"][name]["id"]))

def require_uav(name, label):
    if name not in config["uavs"]:
        raise SystemExit(f"[groundgame] invalid {label}: {name}")

if target_spec in ("", "all"):
    targets = ordered
else:
    targets = [item.strip() for item in target_spec.split(",") if item.strip()]
    for target in targets:
        require_uav(target, "--uav")

for label, value in (("--defender0", defender0), ("--defender1", defender1), ("--enemy", enemy)):
    require_uav(value, label)

if command == "single":
    if len(targets) != 1:
        raise SystemExit("[groundgame] single requires exactly one --uav target")
    if game_role not in ("defender_0", "defender_1", "enemy"):
        raise SystemExit("[groundgame] --role must be defender_0, defender_1, or enemy")

if command == "start" and len({defender0, defender1, enemy}) != 3:
    raise SystemExit("[groundgame] defender0, defender1, and enemy must be three different UAVs")

for key in ("ground_ip",):
    pass

print("GROUND_IP\t" + str(config["ground"]["ip"]))
print("ORDERED\t" + ",".join(ordered))
for target in targets:
    uav = config["uavs"][target]
    print("\t".join(["TARGET", target, str(uav["ssh_host"]), str(uav["project"]), str(uav["ros_master_uri"]), str(uav["ip"])]))
for name in (defender0, defender1, enemy):
    uav = config["uavs"][name]
    print("\t".join(["ROLE", name, str(uav["ssh_host"]), str(uav["project"]), str(uav["ros_master_uri"]), str(uav["ip"])]))
PY
)"
CONFIG_RC=$?
if [ "${CONFIG_RC}" -ne 0 ]; then
  exit "${CONFIG_RC}"
fi
readarray -t CONFIG_ROWS <<<"${CONFIG_OUTPUT}"

GROUND_IP=""
declare -A SSH_HOSTS
declare -A PROJECTS
declare -A ROS_MASTER_URIS
declare -A ROS_IPS
TARGETS=()
ORDERED_UAVS=()

for row in "${CONFIG_ROWS[@]}"; do
  IFS=$'\t' read -r kind a b c d e <<<"${row}"
  case "${kind}" in
    GROUND_IP)
      GROUND_IP="${a}"
      ;;
    ORDERED)
      IFS=',' read -r -a ORDERED_UAVS <<<"${a}"
      ;;
    TARGET)
      TARGETS+=("${a}")
      SSH_HOSTS["${a}"]="${b}"
      PROJECTS["${a}"]="${c}"
      ROS_MASTER_URIS["${a}"]="${d}"
      ROS_IPS["${a}"]="${e}"
      ;;
    ROLE)
      SSH_HOSTS["${a}"]="${b}"
      PROJECTS["${a}"]="${c}"
      ROS_MASTER_URIS["${a}"]="${d}"
      ROS_IPS["${a}"]="${e}"
      ;;
  esac
done

quote() {
  printf "%q" "$1"
}

join_by_comma() {
  local IFS=','
  printf '%s' "$*"
}

single_defender_fleet() {
  local target="$1"
  local role="$2"
  local desired_index=0
  if [ "${role}" = "defender_1" ]; then
    desired_index=1
  fi

  local others=()
  local name
  for name in "${ORDERED_UAVS[@]}"; do
    if [ "${name}" != "${target}" ]; then
      others+=("${name}")
    fi
  done

  local fleet=()
  local other_index=0
  local total="${#ORDERED_UAVS[@]}"
  local idx
  for ((idx = 0; idx < total; idx++)); do
    if [ "${idx}" -eq "${desired_index}" ]; then
      fleet+=("${target}")
    else
      fleet+=("${others[${other_index}]}")
      other_index=$((other_index + 1))
    fi
  done
  join_by_comma "${fleet[@]}"
}

single_enemy_uav() {
  local target="$1"
  local idx
  for ((idx = ${#ORDERED_UAVS[@]} - 1; idx >= 0; idx--)); do
    if [ "${ORDERED_UAVS[${idx}]}" != "${target}" ]; then
      printf '%s' "${ORDERED_UAVS[${idx}]}"
      return 0
    fi
  done
  printf ''
}

remote_prefix() {
  local uav="$1"
  local workspace="$2"
  local project="${PROJECTS[${uav}]}"
  local ros_master_uri="${ROS_MASTER_URIS[${uav}]}"
  local ros_ip="${ROS_IPS[${uav}]}"
  printf 'source /opt/ros/noetic/setup.bash && cd %q && [ -f %q/devel/setup.bash ] && source %q/devel/setup.bash && export UAV_NAME=%q && export ROS_MASTER_URI=%q && export ROS_IP=%q && export ROS_HOSTNAME=%q && unset ROS_NAMESPACE' \
    "${project}" "${workspace}" "${workspace}" "${uav}" "${ros_master_uri}" "${ros_ip}" "${ros_ip}"
}

ssh_run() {
  local uav="$1"
  local command="$2"
  ssh -o BatchMode=yes -o ConnectTimeout=5 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 \
    "${SSH_HOSTS[${uav}]}" bash -lc "$(quote "${command}")"
}

stop_pattern='[r]oslaunch adv adv.launch|[a]dv_ros_adapter|[r]oslaunch mpc mpc.launch|[m]pc_ros_adapter'

remote_stop_command() {
  local force="$1"
  local commands=(
    "set +e"
    "PAT='${stop_pattern}'"
    "pkill -INT -u \"\$(id -u)\" -f \"\$PAT\" >/dev/null 2>&1 || true"
    "sleep 1"
    "pkill -TERM -u \"\$(id -u)\" -f \"\$PAT\" >/dev/null 2>&1 || true"
    "sleep 1"
  )
  if [ "${force}" -eq 1 ]; then
    commands+=("pkill -KILL -u \"\$(id -u)\" -f \"\$PAT\" >/dev/null 2>&1 || true" "sleep 0.5")
  fi
  commands+=(
    "remaining=\$(pgrep -u \"\$(id -u)\" -af \"\$PAT\" || true)"
    "if [ -n \"\$remaining\" ]; then printf '[groundgame] remaining game processes:\\n%s\\n' \"\$remaining\"; exit 2; else printf '[groundgame] stopped game processes\\n'; exit 0; fi"
  )
  local joined
  printf -v joined '%s; ' "${commands[@]}"
  printf '%s' "${joined%; }"
}

run_parallel() {
  local failed=0
  local count=0
  for target in "$@"; do
    (
      case "${COMMAND}" in
        stop)
          printf -- '--- %s stop game ---\n' "${target}"
          ssh_run "${target}" "$(remote_stop_command "${FORCE}")"
          ;;
        status)
          printf -- '--- %s game status ---\n' "${target}"
          ssh_run "${target}" "pgrep -u \"\$(id -u)\" -af '${stop_pattern}' || true"
          ;;
      esac
    ) &
    count=$((count + 1))
    if [ "${count}" -ge "${JOBS}" ]; then
      wait -n || failed=1
      count=$((count - 1))
    fi
  done
  while [ "${count}" -gt 0 ]; do
    wait -n || failed=1
    count=$((count - 1))
  done
  return "${failed}"
}

start_role() {
  local uav="$1"
  local kind="$2"
  local fleet="$3"
  local output_role="${4:-auto}"
  local active_defender_roles="${5:-defender_0,defender_1}"
  local enemy_uav="${6:-${ENEMY}}"
  local log_dir="logs/groundctrl"
  local launch
  if [ "${kind}" = "adv" ]; then
    launch="roslaunch adv adv.launch self_uav:=${uav} fleet_uavs:=${fleet} enemy_uav:=${enemy_uav} output_role:=${output_role} active_defender_roles:=${active_defender_roles} game_end_enabled:=${GAME_END_ENABLED}"
    prefix="$(remote_prefix "${uav}" "adv")"
  else
    launch="roslaunch mpc mpc.launch self_uav:=${uav} fleet_uavs:=${fleet} active_defender_roles:=${active_defender_roles} game_end_enabled:=${GAME_END_ENABLED}"
    prefix="$(remote_prefix "${uav}" "mpc")"
  fi
  local log_suffix="${kind}_${uav}"
  if [ "${COMMAND}" = "single" ]; then
    log_suffix="${kind}_${uav}_${GAME_ROLE}"
  fi
  local launch_inner="nohup ${launch} > ${log_dir}/${log_suffix}.log 2>&1 < /dev/null & pid=\$!; disown \"\$pid\" 2>/dev/null || true; printf '%s\\n' \"\$pid\""
  local command="${prefix} && mkdir -p ${log_dir} && { $(remote_stop_command 0); true; } && bash -lc $(quote "${launch_inner}")"
  printf -- '--- %s start %s ---\n' "${uav}" "${kind}"
  ssh_run "${uav}" "${command}"
}

case "${COMMAND}" in
  start)
    FLEET="${DEFENDER0},${DEFENDER1},${ENEMY}"
    printf '[groundgame] assignment: defender_0=%s defender_1=%s enemy=%s fleet_uavs=%s\n' \
      "${DEFENDER0}" "${DEFENDER1}" "${ENEMY}" "${FLEET}"
    failed=0
    start_role "${DEFENDER0}" adv "${FLEET}" || failed=1
    start_role "${DEFENDER1}" adv "${FLEET}" || failed=1
    start_role "${ENEMY}" mpc "${FLEET}" || failed=1
    exit "${failed}"
    ;;
  single)
    TARGET="${TARGETS[0]}"
    if [ "${GAME_ROLE}" = "enemy" ]; then
      FLEET="$(join_by_comma "${ORDERED_UAVS[@]}")"
      printf '[groundgame] single: %s as enemy, process=mpc fleet_uavs=%s\n' "${TARGET}" "${FLEET}"
      start_role "${TARGET}" mpc "${FLEET}" "enemy" "defender_0,defender_1" ""
    else
      FLEET="$(single_defender_fleet "${TARGET}" "${GAME_ROLE}")"
      ENEMY_FOR_SINGLE="$(single_enemy_uav "${TARGET}")"
      printf '[groundgame] single: %s as %s, process=adv fleet_uavs=%s enemy_uav=%s\n' \
        "${TARGET}" "${GAME_ROLE}" "${FLEET}" "${ENEMY_FOR_SINGLE}"
      start_role "${TARGET}" adv "${FLEET}" "${GAME_ROLE}" "${GAME_ROLE}" "${ENEMY_FOR_SINGLE}"
    fi
    ;;
  stop|status)
    run_parallel "${TARGETS[@]}"
    ;;
esac
