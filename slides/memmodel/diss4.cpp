#include <cstdio>
using namespace std;

enum IntEnum : int {
};

struct IntStruct {
    int i;
};

union IntUnion {
    int i;
};

enum ShortEnum : short {
};

struct ShortStruct {
    short s;
};

union ShortUnion {
    short s;
};

enum CharEnum : char {
};

struct CharStruct {
    char c;
};

union CharUnion {
    char c;
};

template <typename T>
[[gnu::noinline]] void modify(void *p) {
    *(T *)p = T{3};
}

template <typename T>
[[gnu::noinline]] const char *test() {
    int i = 0x10002;
    modify<T>((void *)&i);
    return (i & 0xff) == 3 ? "OK" : "ERROR";
}


int main() {
    printf("以下显示 OK 的就是和 int 兼容的类型\n");
    printf("int:               %s\n", test<int>());
    printf("short:             %s\n", test<short>());
    printf("char:              %s\n", test<char>());
    printf("unsigned int:      %s\n", test<unsigned int>());
    printf("unsigned short:    %s\n", test<unsigned short>());
    printf("unsigned char:     %s\n", test<unsigned char>());
    printf("IntEnum:           %s\n", test<IntEnum>());
    printf("IntStruct:         %s\n", test<IntStruct>());
    printf("IntUnion:          %s\n", test<IntUnion>());
    printf("ShortEnum:         %s\n", test<ShortEnum>());
    printf("ShortStruct:       %s\n", test<ShortStruct>());
    printf("ShortUnion:        %s\n", test<ShortUnion>());
    printf("CharEnum:          %s\n", test<CharEnum>());
    printf("CharStruct:        %s\n", test<CharStruct>());
    printf("CharUnion:         %s\n", test<CharUnion>());
    return 0;
}
