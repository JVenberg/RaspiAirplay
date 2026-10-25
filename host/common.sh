# shellcheck shell=bash disable=SC2034  # sourced by setup.sh and status.sh

HOST_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(dirname "$HOST_DIR")

CARD=sndrpihifiberry
OVERLAY=hifiberry-dacplusadcpro

# "control|value" pairs for the HiFiBerry DAC+ ADC Pro's line input.
MIXER=(
  "ADC Left Input|VINL1[SE]"
  "ADC Right Input|VINR1[SE]"
  "ADC Mic Bias|Mic Bias off"
  "PGA Gain Left|0.0dB"
  "PGA Gain Right|0.0dB"
)

WATCHDOG_FILE=etc/systemd/system.conf.d/90-raspiairplay.conf
JOURNALD_FILE=etc/systemd/journald.conf.d/90-raspiairplay.conf
NM_FILE=etc/NetworkManager/conf.d/90-raspiairplay.conf
DHCPCD_FILE=etc/dhcpcd.exit-hook

os_release() { sed -n "s/^$1=\"\{0,1\}\([^\"]*\)\"\{0,1\}$/\1/p" /etc/os-release; }

boot_config() {
  if [ -f /boot/firmware/config.txt ]; then echo /boot/firmware/config.txt; else echo /boot/config.txt; fi
}

# Loads the HiFiBerry overlay and turns off onboard and HDMI audio so the HiFiBerry is ALSA card 0.
edit_boot_config() {
  sed -i -E \
    -e 's/^dtparam=audio=on/dtparam=audio=off/' \
    -e '/^dtoverlay=vc4-f?kms-v3d/{/noaudio/!s/$/,noaudio/}' \
    "$1"
  if ! grep -qx "dtoverlay=$OVERLAY" "$1"; then
    printf '\n[all]\ndtoverlay=%s\nforce_eeprom_read=0\n' "$OVERLAY" >>"$1"
  fi
}

mixer_get() { amixer -c "$CARD" sget "$1" 2>/dev/null | sed -n "s/^ *Item0: '\(.*\)'$/\1/p"; }

# Matches the repo copy of a config file against the installed one.
installed() { cmp -s "$HOST_DIR/files/$1" "/$1"; }

network_manager() {
  if systemctl is-active -q NetworkManager; then
    echo NetworkManager
  elif systemctl is-active -q dhcpcd; then
    echo dhcpcd
  fi
}
