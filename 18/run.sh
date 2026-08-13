#!/usr/bin/env bash

set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

compiler="${CXX:-g++}"
flags=(
    -std=c++20
    -O3
    -march=native
    -Wall
    -Wextra
    -Wpedantic
)

targets=("$@")
if (( ${#targets[@]} == 0 )); then
    targets=(00 01 02 03 04 05 06)
fi

mkdir -p bin
for requested_target in "${targets[@]}"; do
    target="${requested_target%.cpp}"
    if [[ ! "$target" =~ ^[[:alnum:]_-]+$ ]]; then
        printf 'error: invalid target name: %s\n' "$requested_target" >&2
        exit 1
    fi

    source_file="${target}.cpp"
    output_file="bin/${target}"

    if [[ ! -f "$source_file" ]]; then
        printf 'error: source file not found: %s\n' "$source_file" >&2
        exit 1
    fi

    printf '\n== %s ==\n' "$target"
    "$compiler" "${flags[@]}" "$source_file" -o "$output_file"
    "$output_file"
done
