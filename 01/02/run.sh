g++ -c hello.cpp -o hello.o
g++ -c main.cpp -o main.o
g++ hello.o main.o -o a.out
./a.out
