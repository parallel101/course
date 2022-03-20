set -e

gcc -ffast-math -O3 -fopenmp -fomit-frame-pointer -fverbose-asm -S main.cpp -o /tmp/main.S
vim /tmp/main.S
