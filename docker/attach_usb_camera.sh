#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Attach a Windows USB camera to WSL2 by VID:PID or BUSID using usbipd.

Examples:
  bash docker/attach_usb_camera.sh --id 046d:0893
  bash docker/attach_usb_camera.sh --busid 4-9

Options:
  --id VID:PID     Match device by USB vendor/product ID.
  --busid BUSID    Attach a specific usbipd bus id directly.
  --list           Show usbipd list output and exit.
  -h, --help       Show this help.

Notes:
  - This script must run from WSL and uses powershell.exe to invoke Windows usbipd.
  - If the device is "Not shared", the script will run `usbipd bind --busid ...` first.
  - After attach, verify the camera appeared with `ls /dev/video*`.
EOF
}

if ! command -v powershell.exe >/dev/null 2>&1; then
  echo "powershell.exe was not found. Run this from WSL on Windows." >&2
  exit 1
fi

MODE=""
TARGET=""

while [ $# -gt 0 ]; do
  case "$1" in
    --id)
      MODE="id"
      TARGET="${2:-}"
      shift 2
      ;;
    --busid)
      MODE="busid"
      TARGET="${2:-}"
      shift 2
      ;;
    --list)
      powershell.exe -NoProfile -Command "usbipd list"
      exit 0
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [ -z "${MODE}" ] || [ -z "${TARGET}" ]; then
  usage >&2
  exit 1
fi

TARGET_UPPER=$(printf '%s' "${TARGET}" | tr '[:lower:]' '[:upper:]')

PS_SCRIPT=$(cat <<'EOF'
$ErrorActionPreference = "Stop"
param(
  [string]$Mode,
  [string]$Target
)

$Target = $Target.ToUpperInvariant()
$listOutput = usbipd list | Out-String

if ($Mode -eq "busid") {
  $busid = $Target
  if ($listOutput -notmatch [regex]::Escape($busid)) {
    throw "BUSID '$busid' was not found in `usbipd list`."
  }
} else {
  $matches = [regex]::Matches($listOutput, "^(?<busid>\S+)\s+" + [regex]::Escape($Target) + "\b.*$", [System.Text.RegularExpressions.RegexOptions]::Multiline)
  if ($matches.Count -eq 0) {
    throw "No usbipd device matched VID:PID '$Target'."
  }
  if ($matches.Count -gt 1) {
    $lines = @()
    foreach ($m in $matches) {
      $lines += $m.Groups[0].Value
    }
    throw "Multiple usbipd devices matched VID:PID '$Target':`n$($lines -join "`n")`nUse --busid to choose one explicitly."
  }
  $busid = $matches[0].Groups["busid"].Value
}

$deviceLine = [regex]::Match($listOutput, "^(?<line>" + [regex]::Escape($busid) + ".*)$", [System.Text.RegularExpressions.RegexOptions]::Multiline).Groups["line"].Value
if (-not $deviceLine) {
  throw "Could not locate the full usbipd list line for BUSID '$busid'."
}

Write-Host "Selected: $deviceLine"

if ($deviceLine -match "Not shared") {
  Write-Host "Binding BUSID $busid first..."
  usbipd bind --busid $busid
}

Write-Host "Attaching BUSID $busid to WSL..."
usbipd attach --wsl --busid=$busid
Write-Host "Attach command completed for BUSID $busid."
EOF
)

powershell.exe -NoProfile -Command "${PS_SCRIPT}" -Mode "${MODE}" -Target "${TARGET_UPPER}"

echo
echo "Next checks in WSL:"
echo "  ls /dev/video*"
echo "  v4l2-ctl --list-devices   # if v4l-utils is installed"
