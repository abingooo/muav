#!/usr/bin/env bash
# 若用 sh 调用则自动切换到 bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi
set -euo pipefail
export DISABLE_ROS1_EOL_WARNINGS=1

# 找到仓库根目录，确保相对路径可用
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LANG_APP="$REPO_ROOT/src/langguide/scripts/lang_manage/langmanageapp.py"
REMOTE_SYNC="$REPO_ROOT/src/langguide/scripts/lang_manage/getremotecloud.sh"

cleanup() {
  # 退出时尽量清理后台任务
  if [ -n "${lang_pid:-}" ] && kill -0 "$lang_pid" 2>/dev/null; then
    kill "$lang_pid" 2>/dev/null || true
  fi
  if [ -n "${rqt_pid:-}" ] && kill -0 "$rqt_pid" 2>/dev/null; then
    kill "$rqt_pid" 2>/dev/null || true
  fi
  if [ -n "${remote_pid:-}" ] && kill -0 "$remote_pid" 2>/dev/null; then
    # 同时清理 getremotecloud.sh 启动的子进程（如 clouddisplay.py）
    if command -v pgrep >/dev/null 2>&1; then
      for child in $(pgrep -P "$remote_pid" 2>/dev/null || true); do
        kill "$child" 2>/dev/null || true
      done
    fi
    kill "$remote_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[panel] 启动 langmanageapp (web UI)..."
python3 "$LANG_APP" &
lang_pid=$!
sleep 2
echo "[panel] Web UI: http://localhost:5002"
# 自动打开浏览器（如果可用）
if command -v xdg-open >/dev/null 2>&1; then
  xdg-open "http://localhost:5002" >/dev/null 2>&1 || true
elif command -v open >/dev/null 2>&1; then
  open "http://localhost:5002" >/dev/null 2>&1 || true
else
  echo "[panel] 请手动在浏览器打开 http://localhost:5002"
fi

echo "[panel] 启动远端点云同步脚本..."
sh "$REMOTE_SYNC" &
remote_pid=$!
sleep 2

echo "[panel] 启动 rqt_image_view..."
if command -v rqt_image_view >/dev/null 2>&1; then
  rqt_image_view &
  rqt_pid=$!
  sleep 2
else
  echo "[panel] 未找到 rqt_image_view，请先安装后再手动运行该工具。"
fi

echo "[panel] 启动 rviz..."
roslaunch ego_planner rviz.launch

# roslaunch 退出后，清理后台任务 
cleanup
