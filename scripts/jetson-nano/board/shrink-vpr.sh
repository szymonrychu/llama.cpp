#!/bin/bash
# Run on the Nano as root. Writes /boot/oc.dtb on the SD card: the current custom DTB with the VPR carveout
# shrunk from 400 MB to 32 MB and points the "oc" extlinux entry at it. The VPR region is reusable CMA that
# unmovable allocations (all GPU/nvmap memory) cannot use, so on a headless board it is ~370 MB of RAM the
# GPU never gets. The stock DTB and entries stay untouched.
set -euo pipefail
mount | grep -q " /mnt/sdboot " || { mkdir -p /mnt/sdboot; mount /dev/mmcblk0p1 /mnt/sdboot; }
B=/mnt/sdboot/boot
DTB=$(sed -n "/^LABEL JetsonIO/,/^LABEL/ s#^\s*FDT /boot/##p" $B/extlinux/extlinux.conf | head -1)
dtc -I dtb -O dts -o /tmp/oc.dts "$B/$DTB" 2>/dev/null
python3 - <<'PY'
import re
s = open("/tmp/oc.dts").read()
i = s.index("vpr-carveout {"); j = s.index("};", i)
blk = re.sub(r"size = <0x0+ 0x19000000>;", "size = <0x00 0x02000000>;", s[i:j])
open("/tmp/oc.dts", "w").write(s[:i] + blk + s[j:])
PY
dtc -I dts -O dtb -o $B/oc.dtb /tmp/oc.dts 2>/dev/null
sed -i "/^LABEL oc/,\$ s#FDT /boot/.*#FDT /boot/oc.dtb#" $B/extlinux/extlinux.conf
sed -n "/^LABEL oc/,\$p" $B/extlinux/extlinux.conf | grep -E "LINUX|FDT"
sync; umount /mnt/sdboot
