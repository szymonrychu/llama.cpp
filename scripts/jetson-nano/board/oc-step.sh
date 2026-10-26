#!/bin/bash
# oc-step.sh CPU_KHZ GPU_HZ : pin clocks, then benchmark + correctness probe + deterministic text check.
CPU=$1; GPU=$2
G=/sys/devices/57000000.gpu/devfreq/57000000.gpu
for pass in 1 2; do
  for c in /sys/devices/system/cpu/cpu[0-3]/cpufreq; do echo $CPU | sudo tee $c/scaling_max_freq $c/scaling_min_freq >/dev/null; done
  echo $GPU | sudo tee $G/max_freq $G/min_freq >/dev/null
done
cd /opt/llm/llama.cpp
M=../models/Qwen3VL-2B-Instruct-Q4_K_M.gguf
echo "== cpu $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) gpu $(cat $G/cur_freq)"
./llama-bench -m $M -ngl 99 -fa 1 -ub 256 -p 256 -n 64 -r 2 2>&1 | grep -E "pp256|tg64" | awk -F'|' '{print $(NF-2), $(NF-1)}'
timeout 300 ./test-backend-ops test -b CUDA0 -o MUL_MAT -p "type_a=q[46]_K,type_b=f32,m=(16|4096|1023),n=1," 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep -E "tests passed|FAIL" | head -3
OUT=$(./llama-mtmd-cli -m $M --mmproj ../models/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf -ngl 99 -fa on -p "List the first ten prime numbers and their sum." -n 64 --temp 0 2>/dev/null)
echo "text md5 $(echo "$OUT" | md5sum | cut -c1-12): $(echo "$OUT" | tr '\n' ' ' | tail -c 90)"
echo "temps(AO,CPU,GPU,PLL,PMIC): $(cat /sys/devices/virtual/thermal/thermal_zone[0-4]/temp | tr '\n' ' ')"
echo "kernel errors: $(sudo dmesg | grep -c -E 'gk20a|nvgpu.*(err|fault)|mmu fault|SError|watchdog|soft lockup')"
