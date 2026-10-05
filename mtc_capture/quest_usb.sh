#!/usr/bin/env bash
# Optional: connect a Meta Quest (3 / 3S / Pro) to the capture page over USB instead of Wi-Fi.
# adb forwards the headset's localhost:8012 to this Mac, so the Quest opens
#   https://localhost:8012/?ws=wss://localhost:8012
# No Wi-Fi latency or dropouts, and the Mac's IP no longer matters (the certificate covers localhost).
#
# Once: Quest developer mode on (Meta Horizon phone app > Devices > your Quest > Headset settings >
# Developer mode; needs a Meta developer account), then `brew install --cask android-platform-tools`.
# Each session: plug the Quest into the Mac with a data USB-C cable, put it on and accept
# "Allow USB debugging" (tick "Always allow from this computer"), then run this script.
# A long cable is needed for walking: the whole route segment (~5 m) must be reachable.
#
#   mtc_capture/quest_usb.sh [port]      (default 8012; run it again after re-plugging)
set -euo pipefail
PORT=${1:-8012}
if ! command -v adb >/dev/null; then
    echo "adb not found: brew install --cask android-platform-tools"; exit 1
fi
adb start-server >/dev/null
state=$(adb get-state 2>/dev/null || true)
if [ "$state" != "device" ]; then
    echo "No authorised headset over USB (state: ${state:-none})."
    echo "  - data cable plugged in, headset on, 'Allow USB debugging' accepted inside the headset"
    echo "  - developer mode enabled in the Meta Horizon app"
    adb devices -l
    exit 1
fi
model=$(adb shell getprop ro.product.model 2>/dev/null | tr -d '\r')
adb reverse "tcp:$PORT" "tcp:$PORT"
echo "USB link to $model: the headset's localhost:$PORT -> this Mac's :$PORT"
adb reverse --list
echo
echo "In the Meta Quest Browser open:  https://localhost:$PORT/?ws=wss://localhost:$PORT"
echo "(certificate warning: Advanced > Proceed), then press 'Virtual Reality'."
