set -e

g++ -ffast-math -O3 -fopenmp-simd -fomit-frame-pointer -fverbose-asm -S main.cpp -o /tmp/main.S
#ispc --target=sse2-i32x4 --emit-asm fast.ispc -o /tmp/main.S
vim /tmp/main.S

g++ -ffast-math -O3 -fopenmp-simd main.cpp caller.cpp -o /tmp/a.out
/tmp/a.out
echo -n .;read -n1
