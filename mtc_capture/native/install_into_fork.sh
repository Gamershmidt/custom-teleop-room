#!/usr/bin/env bash
# Copy the MTC capture layer into a checkout of the XRoboToolkit Unity client and let Android
# deliver the capture's UDP beacon:
#   mtc_capture/native/install_into_fork.sh ~/src/XRoboToolkit-Unity-Client
# Re-run after changing anything under mtc_capture/native/unity/; it overwrites Assets/MtcCapture.
set -euo pipefail
if [ $# -ne 1 ] || [ ! -d "$1/Assets" ]; then
    echo "usage: $0 PATH_TO_XRoboToolkit-Unity-Client (the folder that contains Assets/)"; exit 2
fi
FORK=$(cd "$1" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
# update in place, keeping Unity's .meta files (deleting them forces a reimport and can make the first
# compile pass miss newly added scripts); scripts removed here are removed there too
mkdir -p "$FORK/Assets/MtcCapture"
rsync -a --delete --exclude '*.meta' "$HERE/unity/Assets/MtcCapture/" "$FORK/Assets/MtcCapture/"
echo "updated Assets/MtcCapture -> $FORK/Assets/MtcCapture"

MANIFEST="$FORK/Assets/Plugins/Android/AndroidManifest.xml"
PERM='<uses-permission android:name="android.permission.CHANGE_WIFI_MULTICAST_STATE" />'
if [ -f "$MANIFEST" ] && ! grep -q CHANGE_WIFI_MULTICAST_STATE "$MANIFEST"; then
    # after the INTERNET permission, same indentation
    python3 - "$MANIFEST" "$PERM" <<'EOF'
import re, sys
path, perm = sys.argv[1], sys.argv[2]
s = open(path).read()
m = re.search(r'([ \t]*)<uses-permission android:name="android.permission.INTERNET"\s*/>', s)
if not m:
    sys.exit("INTERNET permission not found in " + path + "; add " + perm + " by hand")
s = s[:m.end()] + "\n" + m.group(1) + perm + s[m.end():]
open(path, "w").write(s)
EOF
    echo "added CHANGE_WIFI_MULTICAST_STATE to $MANIFEST"
fi
echo "now open the project in Unity 2022.3.16f1 and build for Android (see mtc_capture/native/README.md)"
