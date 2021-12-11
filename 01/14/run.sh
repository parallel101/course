git submodule update --init --recursive
set -e

rm -rf build
cmake -B build -DCMAKE_TOOLCHAIN_FILE="$PWD/vcpkg/scripts/buildsystems/vcpkg.cmake"
cmake --build build --target a.out

cd vcpkg
sh bootstrap-vcpkg.sh
./vcpkg install fmt:x64-linux
cd ..

build/a.out
