# Jetson Nano board tuning (optional, at your own risk)

These are what the measurements in docs/build-jetson-nano.md were taken with. They overclock past NVIDIA's
limits and raise voltages; the board can degrade or die. Everything is kept side by side with stock: the stock
kernel, DTB and extlinux entries stay, and clocks at boot come from a config file.

Files:
- **`l4t-32.6.1-oc.patch`:** clock tables for L4T 32.6.1. It applies to NVIDIA's `kernel/kernel-4.9` sources.
  It is based on mrcmunir/jetson_nano_overclock (branch jetpack_l4t_32.6.1) and adds:
  - GPU entries at 998.4, 1075.2, 1152, 1228.8 and 1305.6 MHz, max 1300 mV;
  - CPU entries up to 2218.5 MHz, max 1300 mV;
  - removal of the device-tree caps (`nvidia,gpu-max-freq-khz`, `nvidia,dfll-max-freq-khz`).
- **`build-oc-kernel.sh`:** downloads NVIDIA's sources, applies the patch and builds `Image` with the board's
  running config. `LOCALVERSION=-tegra` keeps the existing modules loading.
- **`install-oc-kernel.sh`:** installs `Image.oc` next to the stock `Image` on the SD-card `/boot` and adds a
  DEFAULT `oc` extlinux entry. It also installs `nano-clocks`. With root on a USB SSD, the bootloader still reads
  the SD card's `/boot`.
- **`nano-clocks`, `nano-clocks.default`, `jetson-clocks-override.conf`:** a drop-in for `jetson-clocks.service`.
  It pins CPU and GPU clocks to `/etc/default/nano-clocks`. The default is stock; raise it after testing. A crash
  under an experimental clock reboots into whatever that file says.
- **`oc-step.sh`, `burn-step.sh`:** pin a clock pair, then run llama-bench, a MUL_MAT correctness subset,
  temperatures and kernel-error counts. `burn-step.sh` adds 4 CPU busy-loops for power stress.
- **`shrink-vpr.sh`:** writes `oc.dtb` with the VPR carveout shrunk from 400 MB to 32 MB. That carveout is
  reusable CMA, which unmovable (GPU/nvmap) allocations cannot use. Shrinking it gives about 370 MB back to the
  GPU, which is what lets a 4B model plus its vision projector fit.

Measured on one Nano 4 GB B01 (Qwen3-VL-2B Q4_K_M, tg64 / pp256):

| CPU MHz | GPU MHz | tg64 tok/s | pp256 tok/s | CPU temp °C under load |
|---|---|---|---|---|
| 1479 | 921.6 (stock) | 5.63 | 79.8 | 26 |
| 1479 | 998.4 | 5.99 | 85.1 | 27 |
| 1479 | 1152 | 6.57 | 95.0 | 28 |
| 2014.5 | 1152 (+4 CPU burners) | 6.59 | 94.8 | 45.5 |
| 2014.5 | 1228.8 (+4 CPU burners) | 6.66 | 98.7 | 46 |

- The GPU clock framework caps this chip at 1228.8 MHz, although the table lists 1305.6.
- EMC (memory) stays at 1600 MHz. The Nano has no timing tables above that.

Other memory settings used for the 4B model:
- zram disabled (`systemctl disable nvzramconfig`), so swap goes to disk and actually frees RAM;
- `GGML_CUDA_NO_PINNED=1` and `--no-host`, because pinned host buffers can't be swapped;
- `--cache-ram 0`, because llama-server's host prompt cache otherwise grows until the OOM killer fires.
