# Setting up a Pi

How to take a blank SD card to a working RaspiAirplay box. `host/setup.sh` does the work and is safe to re-run; `host/status.sh` checks the result at any time.

You need a Raspberry Pi 3B+ or newer with a HiFiBerry DAC+ ADC Pro, a spare SD card, and the `.env` file (it is not in git; keep a copy off the Pi).

## 1. Flash the card

In [Raspberry Pi Imager](https://www.raspberrypi.com/software/), pick your Pi model and **Raspberry Pi OS Lite (64-bit)**, then set these under OS customisation:

- Hostname, username and password
- Wi-Fi network, password and country
- Time zone
- SSH on, **public-key authentication only**, with your `~/.ssh/id_ed25519.pub`

Write a spare card when you can, and keep the old one: putting it back is the rollback.

## 2. First boot

Insert the card, power on and give it two minutes. The new install has a new SSH host key, so clear the old one first:

```sh
ssh-keygen -R raspberrypi.local
ssh jackpi@raspberrypi.local
```

## 3. Code and config

On the Pi:

```sh
sudo apt-get install -y git
git clone https://github.com/JVenberg/RaspiAirplay.git
```

From the machine holding `.env`:

```sh
scp .env jackpi@raspberrypi.local:RaspiAirplay/.env
```

## 4. Run setup

```sh
~/RaspiAirplay/host/setup.sh
```

The first run updates the OS, edits the boot config, and installs Docker and Tailscale. Open the Tailscale URL it prints to add the Pi to your tailnet. It then asks for a reboot because the HiFiBerry only appears with the new boot config:

```sh
sudo reboot
~/RaspiAirplay/host/setup.sh   # after reconnecting: sets the mixer, starts the app, runs status.sh
```

In the [Tailscale admin console](https://login.tailscale.com/admin/machines), choose **Disable key expiry** for the Pi, or it drops off the tailnet after 180 days.

## 5. Check

```sh
~/RaspiAirplay/host/status.sh
```

Every line should be `ok`. `WARN` lines are worth reading but don't stop the app; any `FAIL` makes the script exit 1.

## What setup configures

| Setting | Why | Where |
| --- | --- | --- |
| OS packages updated | Security fixes | `apt-get full-upgrade` |
| HiFiBerry overlay loaded; onboard and HDMI audio off | The app records from ALSA `default`, which is card 0, so the HiFiBerry must be the only card | `config.txt` |
| ADC on the line input, Mic Bias off, PGA 0 dB, saved | Mic Bias is for microphones; with a line source it measured no quieter on or off, so off is the safe value | `/var/lib/alsa/asound.state` |
| Hardware watchdog, 15 s | Reboots the Pi if the OS hangs, instead of waiting for someone to pull the plug | [`system.conf.d`](host/files/etc/systemd/system.conf.d/90-raspiairplay.conf) |
| Logs on disk, capped at 500 MB | Raspberry Pi OS keeps logs in RAM, so a power pull would erase the evidence of what went wrong | [`journald.conf.d`](host/files/etc/systemd/journald.conf.d/90-raspiairplay.conf) |
| Wi-Fi power saving off | The Pi's Wi-Fi chip can stall in power-save mode | [NetworkManager](host/files/etc/NetworkManager/conf.d/90-raspiairplay.conf), or a [dhcpcd hook](host/files/etc/dhcpcd.exit-hook) on Debian 11 |
| Docker, from Docker's apt repo | Runs the app with `restart: unless-stopped`, a health check and journald logging (see `docker-compose.yml`) | |
| Tailscale with `--accept-dns=false` | Remote access; leaving DNS alone means a Tailscale outage can't break name resolution | |

Gain stays in `.env` as digital gain (`RASPI_AIRPLAY_GAIN`). Moving it to the ADC's analog PGA measured only 0.3 dB quieter, which isn't worth keeping one setting in two places.

## Day to day

```sh
sudo journalctl -t airplay -f        # app log; a status line every 5 minutes
~/RaspiAirplay/host/status.sh        # health check

cd ~/RaspiAirplay && git pull && sudo docker compose pull && sudo docker compose up -d   # update the app
sudo apt-get update && sudo apt-get full-upgrade && sudo reboot                         # update the OS
```

## The original Debian 11 Pi

Debian 11 is out of support: its packages moved to `archive.debian.org` and get no more fixes, so `status.sh` warns about it. Everything above has been applied to it by hand or by `setup.sh`, so it runs fine until it is reflashed.
