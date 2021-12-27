struct MyVec {
    float x;
    float y;
    float z;
    char padding[4];
};

MyVec a[1024];

void func() {
    for (int i = 0; i < 1024; i++) {
        a[i].x *= a[i].y;
    }
}
