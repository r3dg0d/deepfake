#!/usr/bin/env bash
# Widget launcher for deepfake webcam sessions.
#   run.sh start [virtual]   start a session (virtual: also feed the v4l2loopback virtual cam)
#   run.sh stop               stop the running session (via control.py IPC)
#   run.sh restart [virtual]  stop, wait, start
#   run.sh status             JSON: virtual cam device + whether it's loaded
set -u
dir="$(cd "$(dirname "$0")" && pwd)"
state="${XDG_STATE_HOME:-$HOME/.local/state}/deepfake"
log="$HOME/.cache/deepfake-widget.log"
df="$HOME/.local/bin/deepfake"; [ -x "$df" ] || df="$(command -v deepfake)"
vdev="${DEEPFAKE_VCAM:-}"
if [ -z "$vdev" ]; then for n in /sys/class/video4linux/video*; do
  [ -e "$n/name" ] || continue; case "$(cat "$n/name")" in *"Galaxy Deepfake"*|*Dummy*|*loopback*) vdev="/dev/$(basename "$n")"; break;; esac; done; fi
sid() { python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); print(s.get("session_id","") if s.get("active") else "")' "$state/session.json" 2>/dev/null; }
start() {
  args=(webcam --no-widget)
  if [ "${1:-}" = virtual ]; then
    [ -n "$vdev" ] || { echo "No virtual cam: v4l2loopback isn't loaded (see ~/rice-staging/r3dg0d-virtualcam.nix)"; exit 3; }
    args+=(--output-device "$vdev")
  fi
  [ -n "${DEEPFAKE_DRY:-}" ] && { echo "dry run: deepfake ${args[*]}"; exit 0; }
  mkdir -p "$(dirname "$log")"; echo "== $(date '+%F %r') deepfake ${args[*]}" >>"$log"
  # deepfake's session start calls cleanup_orphans(), which runs `qs kill -p <widget dir>`
  # and kills THIS widget (looked like a crash). The widget is ours here (--no-widget), so
  # point the tool's Quickshell binary at noqs.sh for this session: its kill becomes a no-op.
  DEEPFAKE_QS="$dir/noqs.sh" setsid "$df" "${args[@]}" </dev/null >>"$log" 2>&1 &
  echo "Starting webcam${1:+ + virtual cam ($vdev)}…"
}
stop() { local s; s=$(sid); [ -n "$s" ] || { echo "No active session"; return 0; }; python3 "$dir/control.py" "$s"; }
case "${1:-}" in
  start) start "${2:-}" ;;
  stop) stop ;;
  restart) stop >/dev/null; for _ in $(seq 40); do [ -z "$(sid)" ] && break; sleep 0.25; done; start "${2:-}" ;;
  status) printf '{"vcam":"%s","loaded":%s}\n' "$vdev" "$([ -n "$vdev" ] && echo true || echo false)" ;;
  *) echo "usage: run.sh start [virtual]|stop|restart [virtual]|status" >&2; exit 2 ;;
esac
