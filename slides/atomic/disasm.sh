#!/bin/bash
set -e
K="$1"
F="${2-build/main}"
if [ "x" == "x$K" ]; then
    kg() {
        grep -v '^std::' | grep -v '\.cold$'
    }
else
    kg() {
        local tmp=$RANDOM
        cat > /tmp/.kgtmp.$tmp
        grep "^$K\$" /tmp/.kgtmp.$tmp || grep "^$K" /tmp/.kgtmp.$tmp || grep "$K" /tmp/.kgtmp.$tmp
        rm -f /tmp/.kgtmp.$tmp
    }
fi
objdump -Ct "$F" -j.text | grep -P '^[0-9a-f]{16} [gl]' | sort | awk -F '\t' '{print $2}' | sed 's/[0-9a-f]\{16\}\s\+//' | grep -Pv '^_start$' | kg | xargs -d'\n' -i objdump -Cd "$F" --section=.text --disassembler-color=on -Mintel --disassemble="{}" | grep -Pv '^Disassembly of section ' | grep -Pv "^$F:\\s+file format elf" | grep -Pv '^$' | sed -e 's/\(BYTE\|[A-Z]\{1,3\}WORD\) PTR/\x1b[36m\L\1\x1b[0m/g' | sed -e 's/\(byte\|[a-z]\{1,3\}word\)\x1b\[0m \[/\1\x1b\[0m\[/g' | sed -e 's/\(^[0-9a-f]\{16\} <\)\(.*\)\(>:$\)/\x0a\1\x1b[32m\2\x1b[0m\3/' | sed -e 's/\(# [0-9a-f]\{1,16\} <\)\(.*\)\(>\)/\1\x1b[32m\2\x1b[0m\3/' | tee /tmp/dump.txt
sed -i 's/\x1b\[[0-9]\+m//g' /tmp/dump.txt
# objdump -Ct "$F" -j.bss -j.data -j.rodata | grep -P '^[0-9a-f]{16} [gl]     O ' | grep -Pv '[0-9a-f]              (\.hidden|_IO_)' | grep -v '@GLIBC' | sort | awk -F' ' '{print "\n"$1" <\x1b[32m"$6"\x1b[0m>: "$5}'
