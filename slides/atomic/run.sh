#!/bin/bash

# getopt:
# --file <file>
# --dump <dump>
# --help

FILE=
SYMBOL=
DUMP=false
HELP=false

ARGV0="$0"
while :; do
    case "$1" in
        --file)
            FILE="$2"
            shift 2;;
        --dump-symbol)
            SYMBOL="$2"
            DUMP=true
            shift 2;;
        --dump)
            DUMP=true
            shift;;
        --help)
            HELP=true
            shift;;
        *)
            break;;
    esac
done

if $HELP || [ -z $FILE ]; then
    echo "Usage: $ARGV0 --file <file> [--dump | --dump-symbol <symbol>] [--help]"
    exit 0
fi

CXX=${CXX-c++}
if [ -z $CXXFLAGS ]; then
    CXXFLAGS=(-std=c++20 -O1 -DNDEBUG)
fi

textdump() {
    local OBJ="${1?file}"
    local KEY="$2"
    if [ "x" == "x$KEY" ]; then
        kg() {
            grep -v '^std::' | grep -v '\.cold$' | grep -v '^main$'
        }
    else
        kg() {
            local tmp="$(mktemp)"
            cat > $tmp
            grep "^$KEY\$" $tmp || grep "^$KEY" $tmp || grep "$KEY" $tmp
            rm -f $tmp
        }
    fi
    # objdump -Ct "$OBJ" -j.text
    if [ "xx86_64" == "x$(uname -m)" ]; then
        objdump -Ct "$OBJ" -j.text | grep -P '^[0-9a-f]{16}\s+[glw]' | sort | awk -F '\t' '{print $2}' | sed 's/[0-9a-f]\{16\}\s\+//' | grep -Pv '^\.hidden ' | grep -Pv '^_start(_main)?$' | grep -Pv '^(lib)?unwind[_:]' | grep -Pv 'MTTest::' | kg | xargs -d'\n' -i objdump -Cd "$OBJ" --section=.text --disassembler-color=on -Mintel --disassemble="{}" | grep -Pv '^Disassembly of section ' | grep -Pv "^$OBJ:\\s+file format elf" | grep -Pv '^$' | sed -e 's/\(BYTE\|[A-Z]\{1,3\}WORD\) PTR/\x1b[36m\L\1\x1b[0m/g' | sed -e 's/\(byte\|[a-z]\{1,3\}word\)\x1b\[0m \[/\1\x1b\[0m\[/g' | sed -e 's/\(^[0-9a-f]\{16\} <\)\(.*\)\(>:$\)/\x0a\1\x1b[32m\2\x1b[0m\3/' | sed -e 's/\(# [0-9a-f]\{1,16\} <\)\(.*\)\(>\)/\1\x1b[32m\2\x1b[0m\3/' | sed -e 's/::entry(MTIndex<\([0-9]\+\)ul>)/::entry\1/g' #| tee "$DUMP"
    else
        objdump -Ct "$OBJ" -j.text | grep -P '^[0-9a-f]{16}\s+[glw]' | sort | awk -F '\t' '{print $2}' | sed 's/[0-9a-f]\{16\}\s\+//' | grep -Pv '^\.hidden ' | grep -Pv '^_start(_main)?$' | grep -Pv '^(lib)?unwind[_:]' | grep -Pv 'MTTest::' | kg | xargs -d'\n' -i objdump -Cd "$OBJ" --section=.text --disassembler-color=on --disassemble="{}" | grep -Pv '^Disassembly of section ' | grep -Pv "^$OBJ:\\s+file format elf" | grep -Pv '^$' | sed -e 's/\(^[0-9a-f]\{16\} <\)\(.*\)\(>:$\)/\x0a\1\x1b[32m\2\x1b[0m\3/' | sed -e 's/\(# [0-9a-f]\{1,16\} <\)\(.*\)\(>\)/\1\x1b[32m\2\x1b[0m\3/' | sed -e 's/::entry(MTIndex<\([0-9]\+\)ul>)/::entry\1/g' #| tee "$DUMP"
    fi
    #sed -i 's/\x1b\[[0-9]\+m//g' "$DUMP"
}

OUT="$(mktemp)"
echo "-- Compiling with: $CXX ${CXXFLAGS[@]}"
if $DUMP; then
    oldfile="$FILE"
    FILE="$(mktemp)"
    sed -e 's/^\(\s*\)\(void entry(MT\(Range\)\?Index<[0-9]\+>)\)/\1[[gnu::noinline]] \2/' "$oldfile" > "$FILE"
fi
if "$CXX" "${CXXFLAGS[@]}" "$FILE" -o "$OUT"; then
    echo "-- Testing..."
    "$OUT"
    if $DUMP; then
        textdump "$OUT" "$SYMBOL"
    fi
fi
rm -f "$OUT"
