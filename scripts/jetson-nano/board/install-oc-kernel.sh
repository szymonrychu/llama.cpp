#!/bin/bash
# Install the OC kernel as a separate Image on the SD-card /boot and make it the default extlinux entry.
# The stock Image and the old entries stay; clocks stay at stock until /etc/default/nano-clocks is raised.
set -ex
install -m 755 /tmp/oc/nano-clocks /usr/local/sbin/nano-clocks
[ -f /etc/default/nano-clocks ] || install -m 644 /tmp/oc/nano-clocks.default /etc/default/nano-clocks
mkdir -p /etc/systemd/system/jetson-clocks.service.d
install -m 644 /tmp/oc/jetson-clocks-override.conf /etc/systemd/system/jetson-clocks.service.d/override.conf
systemctl daemon-reload
mkdir -p /mnt/sdboot && mount /dev/mmcblk0p1 /mnt/sdboot
B=/mnt/sdboot/boot
install -m 644 /tmp/oc/Image.oc $B/Image.oc
cp -n $B/extlinux/extlinux.conf $B/extlinux/extlinux.conf.bak-oc
python3 - "$B/extlinux/extlinux.conf" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
if "LABEL oc" not in s:
    m = re.search(r"\nLABEL JetsonIO\n(.*?)(?=\nLABEL |\Z)", s, re.S)
    block = m.group(1).replace("LINUX /boot/Image", "LINUX /boot/Image.oc").replace(
        "MENU LABEL Custom Header Config", "MENU LABEL OC kernel (dvfs tables to GPU 1152 / CPU 2218 MHz) + Custom Header Config")
    s = s.rstrip("\n") + "\n\nLABEL oc\n" + block.rstrip("\n") + "\n"
s = re.sub(r"^DEFAULT .*$", "DEFAULT oc", s, flags=re.M)
open(p, "w").write(s)
PY
grep -n -E "^DEFAULT|^LABEL|LINUX|FDT" $B/extlinux/extlinux.conf
sync; umount /mnt/sdboot
