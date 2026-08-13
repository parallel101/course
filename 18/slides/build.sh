#!/usr/bin/env bash

set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

readonly marp_version="4.5.0"
readonly theme="slides/cache-theme.css"
readonly output_dir="slides/dist"

mkdir -p "$output_dir"

marp=(pnpm dlx "@marp-team/marp-cli@${marp_version}")
common=(slide.md --theme "$theme" --html)

"${marp[@]}" "${common[@]}" -o "$output_dir/cache-set-conflict.html"
"${marp[@]}" "${common[@]}" --browser-path /usr/bin/chromium \
    --pdf -o "$output_dir/cache-set-conflict.pdf"
"${marp[@]}" "${common[@]}" --browser-path /usr/bin/chromium \
    --pptx -o "$output_dir/cache-set-conflict.pptx"
