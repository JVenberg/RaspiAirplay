import subprocess
from pathlib import Path

import pytest

HOST = Path(__file__).resolve().parent.parent / "host"

# Raspberry Pi OS (trixie) stock config.txt, from pi-gen stage1, comments trimmed.
STOCK = """\
#dtparam=i2s=on
dtparam=audio=on
camera_auto_detect=1
display_auto_detect=1
auto_initramfs=1
dtoverlay=vc4-kms-v3d
max_framebuffers=2
disable_fw_kms_setup=1
disable_overscan=1
arm_boost=1

[cm4]
otg_mode=1

[cm5]
dtoverlay=dwc2,dr_mode=host

[pi5]
dtoverlay=nospi10

[all]
"""

# The bullseye Pi this project has run on since 2023.
LEGACY = """\
dtoverlay=hifiberry-dacplusadcpro
force_eeprom_read=0
camera_auto_detect=1
display_auto_detect=1
max_framebuffers=2
arm_64bit=1
disable_overscan=1
[cm4]
otg_mode=1
[all]
[pi4]
arm_boost=1
[all]
"""


def bash(script, *args):
    return subprocess.run(
        ["bash", "-c", f'. "{HOST}/common.sh"; {script}', "bash", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def edit(tmp_path, text):
    path = tmp_path / "config.txt"
    path.write_text(text)
    bash('edit_boot_config "$1"', str(path))
    return path.read_text()


def test_boot_config_from_stock(tmp_path):
    out = edit(tmp_path, STOCK)
    lines = out.splitlines()
    assert "dtparam=audio=off" in lines
    assert "dtparam=audio=on" not in lines
    assert "dtoverlay=vc4-kms-v3d,noaudio" in lines
    assert lines.count("dtoverlay=hifiberry-dacplusadcpro") == 1
    assert lines[-3:] == ["[all]", "dtoverlay=hifiberry-dacplusadcpro", "force_eeprom_read=0"]
    assert out.startswith("#dtparam=i2s=on\n")


def test_boot_config_is_idempotent(tmp_path):
    once = edit(tmp_path, STOCK)
    assert edit(tmp_path, once) == once


def test_boot_config_leaves_configured_pi_alone(tmp_path):
    assert edit(tmp_path, LEGACY) == LEGACY


@pytest.mark.parametrize(
    "line, expected",
    [
        ("dtoverlay=vc4-kms-v3d,cma-256", "dtoverlay=vc4-kms-v3d,cma-256,noaudio"),
        ("dtoverlay=vc4-fkms-v3d", "dtoverlay=vc4-fkms-v3d,noaudio"),
        ("dtoverlay=vc4-kms-v3d,noaudio", "dtoverlay=vc4-kms-v3d,noaudio"),
        ("#dtoverlay=vc4-kms-v3d", "#dtoverlay=vc4-kms-v3d"),
    ],
)
def test_boot_config_silences_hdmi_audio(tmp_path, line, expected):
    assert edit(tmp_path, line + "\n").splitlines()[0] == expected


def test_mixer_get_parses_amixer():
    amixer = """amixer() { cat <<'EOF'
Simple mixer control 'ADC Left Input',0
  Capabilities: enum
  Items: 'No Select' 'VINL1[SE]' 'VINL2[SE]' 'VINL2[SE] + VINL1[SE]' '{VIN1P, VIN1M}[DIFF]'
  Item0: 'VINL1[SE]'
EOF
}"""
    assert bash(f"{amixer}; mixer_get 'ADC Left Input'") == "VINL1[SE]\n"


def test_config_files_exist():
    names = bash('echo "$WATCHDOG_FILE $JOURNALD_FILE $NM_FILE $DHCPCD_FILE"').split()
    assert len(names) == 4
    for name in names:
        assert (HOST / "files" / name).is_file(), name
