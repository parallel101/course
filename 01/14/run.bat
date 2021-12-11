git submodule update --init --recursive

del build
cmake -B build -DCMAKE_TOOLCHAIN_FILE="%CD%/vcpkg/scripts/buildsystems/vcpkg.cmake"
cmake --build build --target a.out

cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg install fmt:x64-linux
cd ..

build\a.out.exe
