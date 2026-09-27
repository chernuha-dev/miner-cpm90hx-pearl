#!/usr/bin/env bash
set -euo pipefail

service=forgeminer-prl.service

if [[ $# -gt 1 || ( $# -eq 1 && ${1:-} != --reboot ) ]]; then
    echo "Usage: sudo $0 [--reboot]" >&2
    exit 2
fi
if [[ ${EUID} -ne 0 ]]; then
    echo "Run as root: sudo $0 ${1:-}" >&2
    exit 1
fi

systemctl daemon-reload
systemctl enable --now "${service}"
systemctl is-enabled --quiet "${service}"
systemctl is-active --quiet "${service}"

echo "${service}: enabled and active; Forge will start after boot."
echo "Current miner: $(systemctl show -P MainPID "${service}")"

if [[ ${1:-} == --reboot ]]; then
    echo "Rebooting now. Check after SSH returns: systemctl is-active ${service}"
    systemctl reboot
fi
