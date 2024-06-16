if grep -q powersave /sys/devices/system/cpu/cpufreq/*/scaling_governor; then
    mode=performance
else
    mode=powersave
fi
echo "Switching to: $mode"
for x in /sys/devices/system/cpu/cpufreq/*/scaling_governor; do sudo sh -c "echo $mode > $x"; done
