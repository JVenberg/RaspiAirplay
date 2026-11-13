#!/usr/bin/env bash
# Brings a Raspberry Pi to the known-good RaspiAirplay setup. Safe to re-run; see SETUP.md.
set -euo pipefail
[ "$(id -u)" -eq 0 ] || exec sudo -- "$0" "$@"
# shellcheck source=host/common.sh
. "$(dirname "$0")/common.sh"

reboot_needed=0
todo=()

step() { printf '\n== %s\n' "$*"; }

# Installs a file from host/files; succeeds only if it changed.
install_file() {
  installed "$1" && return 1
  install -D -m 0644 "$HOST_DIR/files/$1" "/$1"
  echo "installed /$1"
}

step "OS packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get -y -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold full-upgrade
apt-get -y install alsa-utils ca-certificates curl git iw
[ ! -e /run/reboot-required ] || reboot_needed=1

step "Boot config"
cfg=$(boot_config)
before=$(sha256sum <"$cfg")
edit_boot_config "$cfg"
if [ "$(sha256sum <"$cfg")" != "$before" ]; then
  echo "updated $cfg"
  reboot_needed=1
fi

step "Hardware watchdog"
if install_file "$WATCHDOG_FILE"; then systemctl daemon-reexec; fi

step "Persistent, capped logs"
if install_file "$JOURNALD_FILE"; then systemctl restart systemd-journald; fi

step "Wi-Fi power saving off"
case $(network_manager) in
  NetworkManager) install_file "$NM_FILE" || true ;;
  dhcpcd) install_file "$DHCPCD_FILE" || true ;;
esac
if [ -e /sys/class/net/wlan0 ]; then iw dev wlan0 set power_save off; fi

step "Docker"
if ! docker compose version >/dev/null 2>&1; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc]" \
    "https://download.docker.com/linux/debian $(os_release VERSION_CODENAME) stable" \
    >/etc/apt/sources.list.d/docker.list
  apt-get update
  apt-get -y install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
systemctl enable --now docker

step "ADC mixer"
if [ -e "/proc/asound/$CARD" ]; then
  for pair in "${MIXER[@]}"; do
    amixer -q -c "$CARD" sset "${pair%%|*}" "${pair#*|}"
  done
  alsactl store
else
  echo "HiFiBerry not detected; it appears after a reboot with the new boot config."
  reboot_needed=1
fi

step "App"
cd "$REPO_DIR"
if [ ! -f .env ]; then
  todo+=("Create $REPO_DIR/.env (see sample.env), then re-run host/setup.sh.")
elif [ -e "/proc/asound/$CARD" ]; then
  docker compose up -d --pull missing
fi

step "Tailscale"
if ! command -v tailscale >/dev/null; then
  codename=$(os_release VERSION_CODENAME)
  curl -fsSL "https://pkgs.tailscale.com/stable/debian/$codename.noarmor.gpg" \
    -o /usr/share/keyrings/tailscale-archive-keyring.gpg
  curl -fsSL "https://pkgs.tailscale.com/stable/debian/$codename.tailscale-keyring.list" \
    -o /etc/apt/sources.list.d/tailscale.list
  apt-get update
  apt-get -y install tailscale
fi
if ! tailscale status >/dev/null 2>&1; then
  echo "Open the URL below to add this Pi to your tailnet."
  # Leave the Pi's DNS alone so a Tailscale outage can't break name resolution.
  tailscale up --accept-dns=false
fi

[ "$reboot_needed" -eq 0 ] || todo+=("Reboot (sudo reboot), then run host/setup.sh again.")
if [ ${#todo[@]} -gt 0 ]; then
  printf '\nNot done yet:\n'
  printf '  - %s\n' "${todo[@]}"
  exit 1
fi
printf '\nSetup complete.\n'
exec "$HOST_DIR/status.sh"
