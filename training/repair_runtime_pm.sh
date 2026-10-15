#!/usr/bin/env bash
# Disable NVIDIA runtime D3 to test the observed GSP failure on GPU wake.
# Does not stop the desktop, unload drivers, or reboot the machine.
set -euo pipefail

if (( EUID != 0 )); then
    echo 'Run with administrator authentication: sudo bash training/repair_runtime_pm.sh' >&2
    exit 1
fi

gpu=/sys/bus/pci/devices/0000:01:00.0
if [[ ! -r "$gpu/vendor" ]] || [[ "$(cat "$gpu/vendor")" != 0x10de ]]; then
    echo 'Expected NVIDIA device at 0000:01:00.0; stopping.' >&2
    exit 1
fi

config=/etc/modprobe.d/nvidia-runtimepm.conf
rules=/etc/udev/rules.d/99-nvidia-no-runtime-pm.rules
backup=$(mktemp -d /var/backups/nvidia-runtimepm.XXXXXXXX)
for file in "$config" "$rules"; do
    if [[ -e "$file" ]]; then
        cp -a "$file" "$backup/$(basename "$file")"
    else
        touch "$backup/$(basename "$file").absent"
    fi
done
echo "Original configuration recorded in $backup"

# Matching the packaged basename overrides /lib/modprobe.d/nvidia-runtimepm.conf.
cat > "$config" <<'CONFIG'
# Workaround under investigation: GSP boot timeout after runtime D3 resume.
options nvidia NVreg_DynamicPowerManagement=0x00
CONFIG
chmod 0644 "$config"

# Apply after packaged rules, including the PRIME driver's bind rules.
cat > "$rules" <<'RULES'
ACTION=="add|bind", SUBSYSTEM=="pci", ATTR{vendor}=="0x10de", ATTR{class}=="0x030000", TEST=="power/control", ATTR{power/control}="on"
ACTION=="add|bind", SUBSYSTEM=="pci", ATTR{vendor}=="0x10de", ATTR{class}=="0x030200", TEST=="power/control", ATTR{power/control}="on"
RULES
chmod 0644 "$rules"

modprobe -c | awk '$1 == "options" && $2 == "nvidia" { print }'
update-initramfs -u -k "$(uname -r)"
udevadm control --reload-rules
printf 'on\n' > "$gpu/power/control"
echo 'Configuration applied. The loaded driver still needs a new boot.'
echo 'Save your work, fully shut down, then power on and run training/verify_gpu.py.'
echo 'Idle GPU power consumption may increase with runtime D3 disabled.'
