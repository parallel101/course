struct MyVec {
    float x;
    float y;
    float z;
};

MyVec a[1024];

void func() {
    for (int i = 0; i < 1024; i++) {
        a[i].x *= a[i].y;
    }
}
