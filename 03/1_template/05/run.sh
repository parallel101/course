set -e

cmake -B build
cmake --build build --target cpptest
build/cpptest
