struct MyVec {
    float x[1024];
    float y[1024];
    float z[1024];
};

MyVec a;

void func() {
    for (int i = 0; i < 1024; i++) {
        a.x[i] *= a.y[i];
    }
}
