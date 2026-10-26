#!/bin/bash
# burn-step.sh CPU_KHZ GPU_HZ : oc-step under 4 CPU busy-loops (combined power stress).
pids=()
for i in 1 2 3 4; do (while :; do :; done) & pids+=($!); done
sleep 2
bash /opt/llm/oc-step.sh "$1" "$2" | grep -v "text md5"
echo "under burn temps: $(cat /sys/devices/virtual/thermal/thermal_zone[0-4]/temp | tr '\n' ' ') cpu_cur $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq)"
kill "${pids[@]}"
