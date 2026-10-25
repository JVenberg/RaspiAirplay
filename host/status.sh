#!/usr/bin/env bash
# Read-only check of everything host/setup.sh configures. Exits 1 if anything is not ready.
set -uo pipefail
[ "$(id -u)" -eq 0 ] || exec sudo -- "$0" "$@"
# shellcheck source=host/common.sh
. "$(dirname "$0")/common.sh"

fails=0
warns=0
if [ -t 1 ]; then
  G=$'\e[32m' Y=$'\e[33m' R=$'\e[31m' D=$'\e[2m' N=$'\e[0m'
else
  G='' Y='' R='' D='' N=''
fi
ok() { printf '%s ok %s  %s\n' "$G" "$N" "$*"; }
warn() { printf '%sWARN%s  %s\n' "$Y" "$N" "$*"; warns=$((warns + 1)); }
fail() { printf '%sFAIL%s  %s\n' "$R" "$N" "$*"; fails=$((fails + 1)); }
info() { printf '%s      %s%s\n' "$D" "$*" "$N"; }
section() { printf '\n%s\n' "$*"; }

section "System"
info "$(hostname), $(hostname -I | cut -d' ' -f1), $(uptime -p)"
case $(os_release VERSION_CODENAME) in
  buster | bullseye) warn "OS: $(os_release PRETTY_NAME) is out of support; reflash (SETUP.md)" ;;
  *) ok "OS: $(os_release PRETTY_NAME)" ;;
esac

pending=$(apt-get -s -o Debug::NoLocking=1 dist-upgrade 2>/dev/null | grep -c '^Inst ')
if [ "$pending" -eq 0 ]; then
  ok "OS packages up to date (as of the last apt-get update)"
else
  warn "$pending package updates pending: sudo apt-get update && sudo apt-get full-upgrade"
fi

free_gb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
if [ "$free_gb" -ge 2 ]; then ok "Disk: ${free_gb} GB free"; else warn "Disk: only ${free_gb} GB free"; fi

if [ "$(timedatectl show -p NTPSynchronized --value)" = yes ]; then
  ok "Clock synced"
else
  warn "Clock not synced; log times and TLS can be wrong"
fi

if command -v vcgencmd >/dev/null; then
  throttled=$(vcgencmd get_throttled | cut -d= -f2)
  temp=$(vcgencmd measure_temp | cut -d= -f2)
  seen=()
  ((throttled & 0x10000)) && seen+=("under-voltage (use the official power supply)")
  ((throttled & 0x40000)) && seen+=("CPU throttling")
  ((throttled & 0x80000)) && seen+=("soft temperature limit")
  if [ ${#seen[@]} -eq 0 ]; then
    ok "Power and temperature: no problems since boot ($temp)"
  else
    joined=$(printf '%s, ' "${seen[@]}")
    warn "Since boot: ${joined%, } ($throttled, now $temp)"
  fi
fi

section "Audio"
cfg=$(boot_config)
if grep -qx "dtoverlay=$OVERLAY" "$cfg"; then
  ok "Boot config loads $OVERLAY"
else
  fail "$cfg does not load $OVERLAY; run host/setup.sh"
fi

card0=$(sed -n 's/^ *0 \[\([^] ]*\) *\]:.*/\1/p' /proc/asound/cards)
if [ "$card0" = "$CARD" ]; then
  ok "HiFiBerry is ALSA card 0, the app's default device"
elif [ -e "/proc/asound/$CARD" ]; then
  fail "HiFiBerry is not card 0 (card 0 is ${card0:-none}); onboard and HDMI audio must be off"
else
  fail "HiFiBerry not detected; check the HAT and boot config, then reboot"
fi

if [ -e "/proc/asound/$CARD" ]; then
  unsaved=()
  for pair in "${MIXER[@]}"; do
    name=${pair%%|*} want=${pair#*|}
    have=$(mixer_get "$name")
    if [ "$have" = "$want" ]; then
      ok "Mixer: $name = $want"
    elif [[ $name == *Input ]]; then
      fail "Mixer: $name = ${have:-?}, expected $want; the line input is not selected"
    else
      warn "Mixer: $name = ${have:-?}, expected $want"
    fi
    if ! grep -F -A1 "name '$name'" /var/lib/alsa/asound.state 2>/dev/null | grep -qF "value '$have'"; then
      unsaved+=("$name")
    fi
  done
  if [ ${#unsaved[@]} -gt 0 ]; then
    warn "Mixer not saved, so it resets at boot: ${unsaved[*]} (sudo alsactl store)"
  fi

  hw_params=/proc/asound/$CARD/pcm0c/sub0/hw_params
  if grep -q '^rate:' "$hw_params" 2>/dev/null; then
    ok "Capture open: $(awk '/^format:/ {f = $2} /^rate:/ {r = $2} END {print f ", " r " Hz"}' "$hw_params")"
  else
    fail "Capture device is not open; the app is not recording"
  fi
fi

section "Reliability"
watchdog=$(systemctl show -p RuntimeWatchdogUSec --value)
if [ -n "$watchdog" ] && [ "$watchdog" != 0 ]; then
  ok "Hardware watchdog reboots the Pi if it hangs for $watchdog"
else
  fail "Hardware watchdog is off; run host/setup.sh"
fi

if [ -e /sys/class/net/wlan0 ]; then
  power_save=$(iw dev wlan0 get power_save 2>/dev/null | awk '{print $3}')
  case $(network_manager) in
    NetworkManager) persisted=$NM_FILE ;;
    dhcpcd) persisted=$DHCPCD_FILE ;;
    *) persisted='' ;;
  esac
  if [ "$power_save" = off ] && [ -n "$persisted" ] && installed "$persisted"; then
    ok "Wi-Fi power saving off, kept off by /$persisted"
  elif [ "$power_save" = off ]; then
    warn "Wi-Fi power saving is off now but turns back on at boot; run host/setup.sh"
  else
    fail "Wi-Fi power saving is on and can stall the network; run host/setup.sh"
  fi
  signal=$(iw dev wlan0 link 2>/dev/null | awk '/signal:/ {print $2}')
  if [ -z "$signal" ]; then
    fail "Wi-Fi not connected"
  elif [ "$signal" -ge -70 ]; then
    ok "Wi-Fi signal $signal dBm"
  else
    warn "Wi-Fi signal is weak ($signal dBm)"
  fi
fi

usage=$(journalctl --disk-usage 2>/dev/null | grep -oE '[0-9.]+[KMGT]' | head -1)
if [ -z "$(ls -A /var/log/journal 2>/dev/null)" ]; then
  fail "Logs are kept in RAM and lost on power loss; run host/setup.sh"
elif installed "$JOURNALD_FILE"; then
  ok "Logs persist across reboots, capped at 500M (using $usage)"
else
  warn "Logs persist but /$JOURNALD_FILE is missing or differs, so the size cap may not apply"
fi

section "App"
if systemctl is-active -q docker && systemctl is-enabled -q docker; then
  ok "Docker $(docker version -f '{{.Server.Version}}' 2>/dev/null) running and enabled at boot"
else
  fail "Docker is not running or not enabled at boot"
fi

env_file=$REPO_DIR/.env
if [ ! -f "$env_file" ]; then
  fail "$env_file missing; copy sample.env and set RASPI_AIRPLAY_IDENTIFIER"
elif grep -qE '^RASPI_AIRPLAY_IDENTIFIER="?[0-9A-Fa-f]{2}(:[0-9A-Fa-f]{2}){5}"?$' "$env_file"; then
  ok ".env sets the receiver identifier"
else
  warn ".env has no receiver MAC, so the app streams to the first receiver it finds"
fi

container=$(cd "$REPO_DIR" && docker compose ps -aq airplay 2>/dev/null)
if [ -z "$container" ]; then
  fail "App container not created: cd $REPO_DIR && sudo docker compose up -d"
else
  read -r state health restarts policy image < <(docker inspect -f \
    '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}} {{.RestartCount}} {{.HostConfig.RestartPolicy.Name}} {{.Image}}' \
    "$container")
  if [ "$state" = running ] && [ "$health" = healthy ]; then
    ok "App running and healthy (restarts: $restarts, restart policy: $policy)"
  elif [ "$state" = running ] && [ "$health" = starting ]; then
    warn "App starting; health not known yet"
  else
    fail "App is $state, health $health (sudo journalctl -t airplay -n 50)"
  fi
  arch=$(docker image inspect -f '{{.Architecture}}' "$image" 2>/dev/null)
  if [ "$arch" != "$(dpkg --print-architecture)" ]; then
    fail "App image is built for $arch, not $(dpkg --print-architecture)"
  fi
  status=$(journalctl -t airplay --since "-6 min" -o cat --no-pager 2>/dev/null | grep ' status: ' | tail -1)
  if [ -n "$status" ]; then
    ok "Last status: ${status#* status: }"
  else
    warn "No app status line in the last 6 minutes (normal for 5 minutes after a restart)"
  fi
fi

section "Remote access"
if ! command -v tailscale >/dev/null; then
  warn "Tailscale not installed; run host/setup.sh"
elif ! tailscale status >/dev/null 2>&1; then
  warn "Tailscale not logged in: sudo tailscale up --accept-dns=false"
else
  ok "Tailscale connected as $(tailscale ip -4 2>/dev/null | head -1)"
  expiry=$(tailscale status --json --peers=false 2>/dev/null | sed -n 's/.*"KeyExpiry": *"\([^"]*\)".*/\1/p' | head -1)
  if [ -n "$expiry" ]; then
    warn "Tailscale login expires $expiry; disable key expiry for this machine in the admin console"
  fi
fi

if [ "$(sshd -T 2>/dev/null | awk '$1 == "passwordauthentication" {print $2}')" = no ]; then
  ok "SSH accepts keys only"
else
  warn "SSH accepts passwords"
fi

printf '\n%d failed, %d warnings\n' "$fails" "$warns"
[ "$fails" -eq 0 ]
