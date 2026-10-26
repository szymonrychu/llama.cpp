#!/bin/bash
# Build the L4T R32.6.1 kernel Image with the overclock dvfs tables (l4t-32.6.1-oc.patch) inside the
# scripts/jetson-nano Docker image. Usage: build-oc-kernel.sh WORKDIR RUNNING_CONFIG
# RUNNING_CONFIG = `zcat /proc/config.gz` from the Nano. Needs `bc` and `xxd` on PATH (copy them from the Nano
# into WORKDIR/tools if the image lacks them). Output: WORKDIR/out/arch/arm64/boot/Image
set -euo pipefail
W=$(cd "$1" && pwd); CFG=$2; HERE=$(cd "$(dirname "$0")" && pwd)
URL=https://developer.download.nvidia.com/embedded/L4T/r32_Release_v6.1/sources/T210/public_sources.tbz2
cd "$W"
[ -f public_sources.tbz2 ] || wget -q -O public_sources.tbz2 "$URL"
if [ ! -d src ]; then
    tar xjf public_sources.tbz2 Linux_for_Tegra/source/public/kernel_src.tbz2
    mkdir src && tar xjf Linux_for_Tegra/source/public/kernel_src.tbz2 -C src
    (cd src && patch -p1 < "$HERE/l4t-32.6.1-oc.patch")
fi
export PATH="$W/tools:$PATH"
mkdir -p out && cp "$CFG" out/.config
cd src/kernel/kernel-4.9
make O="$W/out" CC=gcc-8 HOSTCC=gcc-8 olddefconfig >/dev/null
# LOCALVERSION must match the running kernel (4.9.253-tegra) or /lib/modules (Wi-Fi etc.) stops loading
make O="$W/out" CC=gcc-8 HOSTCC=gcc-8 LOCALVERSION=-tegra -j"$(nproc)" Image
strings "$W/out/arch/arm64/boot/Image" | grep -m1 "Linux version"
