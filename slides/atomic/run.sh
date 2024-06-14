#!/bin/bash

FILE=${1-main.cpp}

CXX=${CXX-c++}
CXXFLAGS=(-std=c++20 -O3 -DNDEBUG)

OUT="$(mktemp)"

for mode in 1 2 3; do
    "$CXX" "${CXXFLAGS[@]}" -DMODE="$mode" "$FILE" -o "$OUT" && ("$OUT"; dump "$OUT")
    rm -f "$OUT"
done

dump() {
    local OBJ="${1?file}"
    local KEY="$2"
    if [ "x" == "x$KEY" ]; then
        kg() {
            grep -v '^std::' | grep -v '\.cold$'
        }
    else
        kg() {
            local tmp="$(mktemp)"
            cat > $tmp
            grep "^$KEY\$" $tmp || grep "^$KEY" $tmp || grep "$KEY" $tmp
            rm -f $tmp
        }
    fi
    if [ "xx86_64" == "x$(uname -m)" ]; then
        objdump -Ct "$OBJ" -j.text | grep -P '^[0-9a-f]{16} [gl]' | sort | awk -F '\t' '{print $2}' | sed 's/[0-9a-f]\{16\}\s\+//' | grep -Pv '^_start$' | kg | xargs -d'\n' -i objdump -Cd "$OBJ" --section=.text --disassembler-color=on -Mintel --disassemble="{}" | grep -Pv '^Disassembly of section ' | grep -Pv "^$OBJ:\\s+file format elf" | grep -Pv '^$' | sed -e 's/\(BYTE\|[A-Z]\{1,3\}WORD\) PTR/\x1b[36m\L\1\x1b[0m/g' | sed -e 's/\(byte\|[a-z]\{1,3\}word\)\x1b\[0m \[/\1\x1b\[0m\[/g' | sed -e 's/\(^[0-9a-f]\{16\} <\)\(.*\)\(>:$\)/\x0a\1\x1b[32m\2\x1b[0m\3/' | sed -e 's/\(# [0-9a-f]\{1,16\} <\)\(.*\)\(>\)/\1\x1b[32m\2\x1b[0m\3/' | tee "$DUMP"
    else
        objdump -Ct "$OBJ" -j.text | grep -P '^[0-9a-f]{16} [gl]' | sort | awk -F '\t' '{print $2}' | sed 's/[0-9a-f]\{16\}\s\+//' | grep -Pv '^_start$' | kg | xargs -d'\n' -i objdump -Cd "$OBJ" --section=.text --disassembler-color=on --disassemble="{}" | grep -Pv '^Disassembly of section ' | grep -Pv "^$OBJ:\\s+file format elf" | grep -Pv '^$' | sed -e 's/\(^[0-9a-f]\{16\} <\)\(.*\)\(>:$\)/\x0a\1\x1b[32m\2\x1b[0m\3/' | sed -e 's/\(# [0-9a-f]\{1,16\} <\)\(.*\)\(>\)/\1\x1b[32m\2\x1b[0m\3/' | tee "$DUMP"
    fi
    sed -i 's/\x1b\[[0-9]\+m//g' "$DUMP"
}
