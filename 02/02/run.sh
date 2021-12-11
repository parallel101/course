set -e

cmake -B build
cmake --build build
build/legacy
build/modern
